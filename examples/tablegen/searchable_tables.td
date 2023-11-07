// This example uses the Searchable Tables backend, aka --gen-searchable-tables.
// To choose the backend, select it as the "Action" in the "Overrides" menu,
// or add "--gen-searchable-tables" to the compiler options.
// For more details:
// https://llvm.org/docs/TableGen/BackEnds.html#searchabletables-reference

include "llvm/TableGen/SearchableTable.td"

class Pair<string name, int value> : SearchableTable {
  let SearchableFields = ["Name", "Value"];
  let EnumValueField = "Value";

  string Name = name;
  bits<2> Value = value;
}

def : Pair<"Example 0", 0b00>;
def : Pair<"Example 1", 0b01>;
def : Pair<"Example 2", 0b10>;
def : Pair<"Example 3", 0b11>;