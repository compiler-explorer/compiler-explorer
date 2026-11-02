#ifdef GET_PAIR_DECL
enum PairValues {
  Example 0 = 0,
  Example 1 = 1,
  Example 2 = 2,
  Example 3 = 3,
};
#endif

#ifdef GET_PAIR_DECL
const Pair *lookupPairByName(StringRef Name);
const Pair *lookupPairByValue(uint8_t Value);
StringRef getPairStr(StringTable::Offset);
#endif

#ifdef GET_PAIR_IMPL
constexpr Pair PairsList[] = {
  { 1 /* "Example 0" */, 0x0 }, // 0
  { 11 /* "Example 1" */, 0x1 }, // 1
  { 21 /* "Example 2" */, 0x2 }, // 2
  { 31 /* "Example 3" */, 0x3 }, // 3
 };

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif
static constexpr char PairsListStringsStorage[] =
  "\0"
  "Example 0\0"
  "Example 1\0"
  "Example 2\0"
  "Example 3\0"
  "EXAMPLE 0\0"
  "EXAMPLE 1\0"
  "EXAMPLE 2\0"
  "EXAMPLE 3\0"
  ;
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

static constexpr llvm::StringTable
PairsListStrings = PairsListStringsStorage;

StringRef getPairStr(StringTable::Offset Offset) {
  return PairsListStrings[Offset];
}

const Pair *lookupPairByName(StringRef Name) {
  struct IndexType {
    unsigned Name;
    unsigned _index;
  };
  static const struct IndexType Index[] = {
    { 41 /* "EXAMPLE 0" */, 0 },
    { 51 /* "EXAMPLE 1" */, 1 },
    { 61 /* "EXAMPLE 2" */, 2 },
    { 71 /* "EXAMPLE 3" */, 3 },
  };

  struct KeyType {
    std::string Name;
  };
  KeyType Key = {Name.upper()};
  struct Comp {
    bool operator()(const IndexType &LHS, const KeyType &RHS) const {
      StringRef LHSNameStr = PairsListStrings[LHS.Name];
      StringRef RHSNameStr = RHS.Name;
      int CmpName = LHSNameStr.compare(RHSNameStr);
      if (CmpName < 0) return true;
      if (CmpName > 0) return false;
      return false;
    }
  };
  auto Table = ArrayRef(Index);
  auto Idx = std::lower_bound(Table.begin(), Table.end(), Key, Comp());
  if (Idx == Table.end() ||
      Key.Name != PairsListStrings[Idx->Name])
    return nullptr;

  return &PairsList[Idx->_index];
}

const Pair *lookupPairByValue(uint8_t Value) {
  struct IndexType {
    uint8_t Value;
    unsigned _index;
  };
  static const struct IndexType Index[] = {
    { 0x0, 0 },
    { 0x1, 1 },
    { 0x2, 2 },
    { 0x3, 3 },
  };

  if ((uint8_t)Value != std::clamp<uint8_t>(Value, 0x0, 0x3))
    return nullptr;

  auto Table = ArrayRef(Index);
  size_t Idx = Value - 0x0;
  return &PairsList[Table[Idx]._index];
}
#endif

#undef GET_PAIR_DECL
#undef GET_PAIR_IMPL
