Compiled from "example.java"
class Main {
  Main();
    Code:
       0: aload_0
       1: invokespecial #1                  // Method java/lang/Object."<init>":()V
       4: return
    LineNumberTable:
      line 1: 0

  static void test(int);
    Code:
       0: iload_0
       1: lookupswitch  { // 2
                     0: 28
                     1: 38
               default: 49
          }
      28: getstatic     #7                  // Field java/lang/System.out:Ljava/io/PrintStream;
      31: iconst_1
      32: invokevirtual #13                 // Method java/io/PrintStream.println:(I)V
      35: goto          57
      38: getstatic     #7                  // Field java/lang/System.out:Ljava/io/PrintStream;
      41: bipush        12
      43: invokevirtual #13                 // Method java/io/PrintStream.println:(I)V
      46: goto          57
      49: getstatic     #7                  // Field java/lang/System.out:Ljava/io/PrintStream;
      52: ldc           #19                 // String default
      54: invokevirtual #21                 // Method java/io/PrintStream.println:(Ljava/lang/String;)V
      57: return
    LineNumberTable:
      line 3: 0
      line 4: 28
      line 5: 38
      line 6: 49
      line 8: 57
}
