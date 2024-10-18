// Write your C program in this section:
int main(){ return 0; }

// Write your Coccinelle 🐞 rules in this section (and keep the ifdef):
#ifdef COCCINELLE // this keyword should only occur on this and the last line
@patch@
@@
- 0
+ 1
#endif /* COCCINELLE */
