// Write your C program in this section:
int main(){ return 0; }

// Write your Coccinelle 🐞 rules in this section (and keep the ifdef):
#ifdef COCCINELLE // this keyword should only occur on this and the last line
@patch@
constant c;
@@
// finds a constant and replaces with 1
- c
+ 1

@@
constant patch.c;
@@
// inserts last rule's matched constant (0) next to any occurrence of 1
- 1
+ 1 + c
#endif /* COCCINELLE */
