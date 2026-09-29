"""Minimal read-only XLSX dumper used when no workbook runtime is available."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def text(node):
    return "".join(part.text or "" for part in node.iter(f"{{{MAIN}}}t"))


def main(path: str):
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    end = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
    sheet_filter = sys.argv[4] if len(sys.argv) > 4 else None
    with zipfile.ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [text(item) for item in root.findall(f"{{{MAIN}}}si")]
        rel_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relations = {
            item.attrib["Id"]: item.attrib["Target"]
            for item in rel_root.findall(f"{{{PKG_REL}}}Relationship")
        }
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        for sheet in workbook.iter(f"{{{MAIN}}}sheet"):
            if sheet_filter and sheet.attrib["name"] != sheet_filter:
                continue
            target = relations[sheet.attrib[f"{{{REL}}}id"]].lstrip("/")
            if not target.startswith("xl/"):
                target = "xl/" + target
            root = ET.fromstring(archive.read(target))
            print(f"\n### {sheet.attrib['name']}")
            for row in root.iter(f"{{{MAIN}}}row"):
                row_number = int(row.attrib.get("r", "0"))
                if not start <= row_number <= end:
                    continue
                values = []
                for cell in row.findall(f"{{{MAIN}}}c"):
                    kind = cell.attrib.get("t")
                    value_node = cell.find(f"{{{MAIN}}}v")
                    inline = cell.find(f"{{{MAIN}}}is")
                    formula = cell.find(f"{{{MAIN}}}f")
                    value = ""
                    if inline is not None:
                        value = text(inline)
                    elif value_node is not None:
                        value = value_node.text or ""
                        if kind == "s":
                            value = shared[int(value)]
                        elif kind == "b":
                            value = "TRUE" if value == "1" else "FALSE"
                    if formula is not None:
                        value = f"={formula.text} -> {value}"
                    if value != "":
                        values.append(f"{cell.attrib.get('r')}={value!r}")
                if values:
                    print(" | ".join(values))


if __name__ == "__main__":
    main(sys.argv[1])
