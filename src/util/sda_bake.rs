//! Bakes `r13`-relative small-data accesses into compiled objects.
//!
//! Some games (e.g. those linked with SN Systems' `ngcld`) address *all* small data through `r13`, including `.sdata2`/`.sbss2`.
//! `mwld` always resolves `R_PPC_EMB_SDA21` relocations against those sections via `r2`, so compiled code referencing them fails to link/match.
//! Fixed displacements would pin every such access to the original layout. Instead, the access survives the link as an `R_PPC_ADDR16_LO` relocation on an `r13`-based instruction.
//! `mwld` resolves this without a range check:
//! - `dtk dol split` converts split objects' conflicting relocations, and records which `.sdata2`/`.sbss2` symbols exist so that compiled objects' references to them can be recognised
//! - `dtk elf bake-sda` converts compiled objects' relocations before the link
//! - `dtk elf resolve-sda` rewrites each one in the linked executable as `symbol + addend - _SDA_BASE_`, from the final layout, and turns the relocation back into `R_PPC_EMB_SDA21`

use std::collections::BTreeMap;

use anyhow::{Result, anyhow, bail, ensure};
use object::elf;
use serde::{Deserialize, Serialize};

use crate::obj::ObjInfo;

/// Sections `mwld` would resolve via `r2`.
const SDA2_SECTIONS: [&str; 2] = [".sdata2", ".sbss2"];

fn is_sda2_section(name: &str) -> bool {
    SDA2_SECTIONS.contains(&name.split(':').next().unwrap_or(name))
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SdaBakeData {
    /// Value of `_SDA_BASE_` (`r13`) in the original binary. Informational: nothing is baked against it.
    pub sda_base: u32,
    /// Every global symbol located in `.sdata2`/`.sbss2`, by name, with its address in the original binary.
    /// `bake-sda` only uses the names, to recognise references to undefined symbols.
    pub symbols: BTreeMap<String, u32>,
}

impl SdaBakeData {
    pub fn from_obj(obj: &ObjInfo) -> Result<Self> {
        let sda_base = obj.sda_base.ok_or_else(|| anyhow!("_SDA_BASE_ is unknown"))?;
        let mut data = SdaBakeData { sda_base, ..Default::default() };
        for (section_index, section) in obj.sections.iter() {
            if !is_sda2_section(&section.name) {
                continue;
            }
            for (_, symbol) in obj.symbols.for_section(section_index) {
                // Only globals can be referenced from other objects
                if symbol.name.is_empty() || symbol.flags.is_local() {
                    continue;
                }
                data.symbols.insert(symbol.name.clone(), symbol.address as u32);
            }
        }
        Ok(data)
    }
}

struct SectionHeader {
    name: String,
    kind: u32,
    addr: u32,
    offset: usize,
    size: usize,
    link: u32,
    info: u32,
    entsize: usize,
}

fn read_u16(data: &[u8], offset: usize) -> Result<u16> {
    let bytes = data.get(offset..offset + 2).ok_or_else(|| anyhow!("Truncated ELF"))?;
    Ok(u16::from_be_bytes(bytes.try_into().unwrap()))
}

fn read_u32(data: &[u8], offset: usize) -> Result<u32> {
    let bytes = data.get(offset..offset + 4).ok_or_else(|| anyhow!("Truncated ELF"))?;
    Ok(u32::from_be_bytes(bytes.try_into().unwrap()))
}

fn write_u32(data: &mut [u8], offset: usize, value: u32) {
    data[offset..offset + 4].copy_from_slice(&value.to_be_bytes());
}

fn read_str(data: &[u8], offset: usize) -> Result<String> {
    let bytes = data.get(offset..).ok_or_else(|| anyhow!("Truncated ELF"))?;
    let end = bytes.iter().position(|&b| b == 0).ok_or_else(|| anyhow!("Unterminated string"))?;
    Ok(String::from_utf8_lossy(&bytes[..end]).into_owned())
}

fn read_section_headers(data: &[u8], e_type: u16) -> Result<(usize, usize, Vec<SectionHeader>)> {
    ensure!(data.get(..4) == Some(&elf::ELFMAG[..]), "Not an ELF file");
    ensure!(
        data.get(4) == Some(&elf::ELFCLASS32) && data.get(5) == Some(&elf::ELFDATA2MSB),
        "Expected a 32-bit big-endian ELF"
    );
    ensure!(
        read_u16(data, 16)? == e_type,
        "Expected {}",
        if e_type == elf::ET_REL { "a relocatable object" } else { "an executable" }
    );
    let shoff = read_u32(data, 32)? as usize;
    let shentsize = read_u16(data, 46)? as usize;
    let shnum = read_u16(data, 48)? as usize;
    let shstrndx = read_u16(data, 50)? as usize;
    ensure!(shentsize >= 40, "Invalid section header size");
    let mut headers = Vec::with_capacity(shnum);
    let mut name_offsets = Vec::with_capacity(shnum);
    for i in 0..shnum {
        let base = shoff + i * shentsize;
        name_offsets.push(read_u32(data, base)? as usize);
        headers.push(SectionHeader {
            name: String::new(),
            kind: read_u32(data, base + 4)?,
            addr: read_u32(data, base + 12)?,
            offset: read_u32(data, base + 16)? as usize,
            size: read_u32(data, base + 20)? as usize,
            link: read_u32(data, base + 24)?,
            info: read_u32(data, base + 28)?,
            entsize: read_u32(data, base + 36)? as usize,
        });
    }
    let strtab_offset =
        headers.get(shstrndx).ok_or_else(|| anyhow!("Invalid section name table"))?.offset;
    for (header, name_offset) in headers.iter_mut().zip(name_offsets) {
        header.name = read_str(data, strtab_offset + name_offset)?;
    }
    Ok((shoff, shentsize, headers))
}

/// Whether `ins` is a D-form instruction whose 16-bit displacement a small-data relocation can address.
fn is_sda_dform(ins: u32) -> bool {
    let op = ins >> 26;
    op == 14 || (32..=55).contains(&op)
}

/// Converts every `R_PPC_EMB_SDA21` relocation targeting `.sdata2`/`.sbss2` into `R_PPC_ADDR16_LO` relocation with `r13` instruction.
/// Returns the number of relocations converted.
pub fn bake_object(data: &mut [u8], bake: &SdaBakeData) -> Result<usize> {
    let (_, _, headers) = read_section_headers(data, elf::ET_REL)?;
    let mut baked = 0;
    for rela in headers.iter() {
        if rela.kind != elf::SHT_RELA || rela.size == 0 {
            continue;
        }
        let entsize = if rela.entsize == 0 { 12 } else { rela.entsize };
        ensure!(entsize >= 12 && rela.size % entsize == 0, "Invalid relocation section");
        let target = headers
            .get(rela.info as usize)
            .ok_or_else(|| anyhow!("Invalid relocation target section"))?;
        let symtab =
            headers.get(rela.link as usize).ok_or_else(|| anyhow!("Invalid symbol table"))?;
        let strtab = headers
            .get(symtab.link as usize)
            .ok_or_else(|| anyhow!("Invalid string table"))?
            .offset;
        let sym_entsize = if symtab.entsize == 0 { 16 } else { symtab.entsize };

        for entry_offset in (rela.offset..rela.offset + rela.size).step_by(entsize) {
            let r_offset = read_u32(data, entry_offset)?;
            let r_info = read_u32(data, entry_offset + 4)?;
            if r_info & 0xFF != elf::R_PPC_EMB_SDA21 {
                continue;
            }
            let sym = symtab.offset + (r_info >> 8) as usize * sym_entsize;
            let st_name = read_u32(data, sym)? as usize;
            let st_shndx = read_u16(data, sym + 14)?;
            let name = read_str(data, strtab + st_name)?;
            let is_sda2 = if st_shndx == elf::SHN_UNDEF {
                bake.symbols.contains_key(&name)
            } else {
                headers.get(st_shndx as usize).is_some_and(|section| is_sda2_section(&section.name))
            };
            if !is_sda2 {
                continue;
            }
            // mwcc points the relocation at the low halfword of the instruction
            let ins_offset = target.offset + (r_offset & !3) as usize;
            let ins = read_u32(data, ins_offset)?;
            ensure!(
                is_sda_dform(ins),
                "Small-data relocation on '{}' at {:#X} is not on a D-form instruction ({:#010X})",
                name,
                r_offset,
                ins
            );
            write_u32(data, ins_offset, (ins & !0x1F_FFFF) | (13 << 16));
            write_u32(data, entry_offset, (r_offset & !3) + 2);
            write_u32(data, entry_offset + 4, (r_info & !0xFF) | elf::R_PPC_ADDR16_LO);
            baked += 1;
        }
    }
    Ok(baked)
}

/// Resolves the converted small-data accesses in the linked executable `data`.
/// Nothing else uses that combination: a real `@l` half never goes through `r13`, which is reserved.
/// Returns the number of accesses resolved.
pub fn resolve_executable(data: &mut [u8]) -> Result<usize> {
    let (_, _, headers) = read_section_headers(data, elf::ET_EXEC)?;
    let symtab = headers
        .iter()
        .find(|h| h.kind == elf::SHT_SYMTAB)
        .ok_or_else(|| anyhow!("No symbol table: link without stripping symbols"))?;
    let strtab =
        headers.get(symtab.link as usize).ok_or_else(|| anyhow!("Invalid string table"))?.offset;
    let sym_entsize = if symtab.entsize == 0 { 16 } else { symtab.entsize };
    let mut sda_base = None;
    for sym in (symtab.offset..symtab.offset + symtab.size).step_by(sym_entsize) {
        if read_str(data, strtab + read_u32(data, sym)? as usize)? == "_SDA_BASE_" {
            sda_base = Some(read_u32(data, sym + 4)?);
            break;
        }
    }
    let sda_base = sda_base.ok_or_else(|| anyhow!("_SDA_BASE_ is not defined"))?;

    let mut resolved = 0;
    for rela in headers.iter() {
        if rela.kind != elf::SHT_RELA || rela.size == 0 {
            continue;
        }
        let entsize = if rela.entsize == 0 { 12 } else { rela.entsize };
        ensure!(entsize >= 12 && rela.size % entsize == 0, "Invalid relocation section");
        let target = headers
            .get(rela.info as usize)
            .ok_or_else(|| anyhow!("Invalid relocation target section"))?;
        if target.kind != elf::SHT_PROGBITS {
            continue;
        }
        for entry_offset in (rela.offset..rela.offset + rela.size).step_by(entsize) {
            let r_offset = read_u32(data, entry_offset)?;
            let r_info = read_u32(data, entry_offset + 4)?;
            if r_info & 0xFF != elf::R_PPC_ADDR16_LO {
                continue;
            }
            let ins_addr = r_offset & !3;
            let Some(rel) =
                ins_addr.checked_sub(target.addr).filter(|&o| (o as usize) < target.size)
            else {
                continue;
            };
            let ins_offset = target.offset + rel as usize;
            let ins = read_u32(data, ins_offset)?;
            if !is_sda_dform(ins) || (ins >> 16) & 0x1F != 13 {
                continue;
            }
            let sym = symtab.offset + (r_info >> 8) as usize * sym_entsize;
            let st_value = read_u32(data, sym + 4)?;
            let st_shndx = read_u16(data, sym + 14)?;
            if !headers.get(st_shndx as usize).is_some_and(|s| is_sda2_section(&s.name)) {
                continue;
            }
            let name = read_str(data, strtab + read_u32(data, sym)? as usize)?;
            let address = st_value.wrapping_add(read_u32(data, entry_offset + 8)?);
            ensure!(
                ins & 0xFFFF == address & 0xFFFF,
                "'{}' at {:#010X}: the linker wrote {:#06X}, expected the low half of {:#010X}",
                name,
                ins_addr,
                ins & 0xFFFF,
                address
            );
            let disp = address.wrapping_sub(sda_base) as i32;
            let Ok(disp) = i16::try_from(disp) else {
                bail!(
                    "'{}' at {:#010X} is out of range of _SDA_BASE_ ({:#010X}), accessed at {:#010X}",
                    name,
                    address,
                    sda_base,
                    ins_addr
                );
            };
            write_u32(data, ins_offset, (ins & !0xFFFF) | disp as u16 as u32);
            write_u32(data, entry_offset, ins_addr);
            write_u32(data, entry_offset + 4, (r_info & !0xFF) | elf::R_PPC_EMB_SDA21);
            resolved += 1;
        }
    }
    Ok(resolved)
}

#[cfg(test)]
mod test {
    use object::{
        Architecture, BinaryFormat, Endianness, RelocationFlags, SectionKind, SymbolFlags,
        SymbolKind, SymbolScope,
        write::{Object, Relocation, Symbol, SymbolSection},
    };

    use super::*;

    #[test]
    fn bake_converts_only_sda2_relocations() {
        let mut obj = Object::new(BinaryFormat::Elf, Architecture::PowerPc, Endianness::Big);
        let text = obj.add_section(vec![], b".text".to_vec(), SectionKind::Text);
        // lwz r3, 0(r0) ×3, lfs f1, 0(r0)
        obj.append_section_data(
            text,
            &[
                0x80, 0x60, 0x00, 0x00, 0x80, 0x60, 0x00, 0x00, 0x80, 0x60, 0x00, 0x00, 0xC0, 0x20,
                0x00, 0x00,
            ],
            4,
        );
        let sdata2 = obj.add_section(vec![], b".sdata2".to_vec(), SectionKind::ReadOnlyData);
        obj.append_section_data(sdata2, &[0; 8], 4);
        let extern_sym = |name: &str| Symbol {
            name: name.as_bytes().to_vec(),
            value: 0,
            size: 0,
            kind: SymbolKind::Data,
            scope: SymbolScope::Dynamic,
            weak: false,
            section: SymbolSection::Undefined,
            flags: SymbolFlags::None,
        };
        let gx_data = obj.add_symbol(extern_sym("__GXData"));
        let sdata_var = obj.add_symbol(extern_sym("someSdataVar"));
        let float_const = obj.add_symbol(Symbol {
            name: b"@123".to_vec(),
            value: 4,
            size: 4,
            kind: SymbolKind::Data,
            scope: SymbolScope::Compilation,
            weak: false,
            section: SymbolSection::Section(sdata2),
            flags: SymbolFlags::None,
        });
        let sda21 = |offset, symbol, addend| Relocation {
            offset,
            symbol,
            addend,
            flags: RelocationFlags::Elf { r_type: elf::R_PPC_EMB_SDA21 },
        };
        // mwcc-style offset (instruction + 2)
        obj.add_relocation(text, sda21(2, gx_data, 0)).unwrap();
        obj.add_relocation(text, sda21(4, sdata_var, 0)).unwrap();
        obj.add_relocation(text, sda21(8, gx_data, 4)).unwrap();
        obj.add_relocation(text, sda21(14, float_const, 0)).unwrap();
        let mut data = obj.write().unwrap();

        let bake = SdaBakeData {
            sda_base: 0x806734E0,
            symbols: BTreeMap::from([("__GXData".to_string(), 0x80670788)]),
        };
        assert_eq!(bake_object(&mut data, &bake).unwrap(), 3);

        let file = object::File::parse(&*data).unwrap();
        use object::{Object as _, ObjectSection as _};
        let text = file.section_by_name(".text").unwrap();
        let bytes = text.data().unwrap();
        let ins = |i: usize| u32::from_be_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
        // .sdata2 accesses are now r13-based, the .sdata access is untouched
        assert_eq!(ins(0), 0x806D0000);
        assert_eq!(ins(1), 0x80600000);
        assert_eq!(ins(2), 0x806D0000);
        assert_eq!(ins(3), 0xC02D0000);
        // Every relocation is kept; the .sdata2 ones are ADDR16_LO on the low halfword, with their addends
        let relocs: Vec<_> = text
            .relocations()
            .map(|(offset, r)| match r.flags() {
                RelocationFlags::Elf { r_type } => (offset, r_type, r.addend()),
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(relocs, vec![
            (2, elf::R_PPC_ADDR16_LO, 0),
            (4, elf::R_PPC_EMB_SDA21, 0),
            (10, elf::R_PPC_ADDR16_LO, 4),
            (14, elf::R_PPC_ADDR16_LO, 0),
        ]);
    }
}
