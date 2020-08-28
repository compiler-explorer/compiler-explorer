// from https://github.com/compiler-explorer/compiler-explorer/issues/1081
int switchexample(unsigned char num) {
   switch(num) {
   case 0:
        return 123;
   case 2:
        return 124;
   case 4:
        return 125;
   case 6:
        return 126;
   case 8:
        return 127;
   case 10:
        return 128;
   case 12:
        return 129;
   case 14:
        return 130;
   case 16:
        return 131;
   case 18:
        return 132;
   case 20:
        return 133;
   case 22:
        return 134;
   case 24: {
        volatile char asdf = num * num;
        return 145;
   }
   case 125:
        return 2;
   case 126:
        return 3;
   case 127:
        return 3;
   case 128:
        return 4;
   case 137:
        return 146;
   case 138:
        return 147;
   case 139:
        return 148;
   case 255:
        return 149;
    default:
        return 1;
   }
}
