// Write your C++ program in this section:
int main(){ return 0; }

#ifdef COCCINELLE // this keyword should only occur on this and the last line
// Write your Coccinelle 🐞 rules in this section (and keep the ifdef):
@patch@
@@
- 0
+ 1
#endif /* COCCINELLE */
