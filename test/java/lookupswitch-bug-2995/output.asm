class Main {
  Main();
line 1:       0: aload_0
line 1:       1: invokespecial #1                  // Method java/lang/Object."<init>":()V
line 1:       4: return


  static void test(int);
line 3:       0: iload_0
line 3:       1: lookupswitch  { // 2
line 3:                     0: 28
line 3:                     1: 38
line 3:               default: 49
line 3:          }
line 4:      28: getstatic     #7                  // Field java/lang/System.out:Ljava/io/PrintStream;
line 4:      31: iconst_1
line 4:      32: invokevirtual #13                 // Method java/io/PrintStream.println:(I)V
line 4:      35: goto          57
line 5:      38: getstatic     #7                  // Field java/lang/System.out:Ljava/io/PrintStream;
line 5:      41: bipush        12
line 5:      43: invokevirtual #13                 // Method java/io/PrintStream.println:(I)V
line 5:      46: goto          57
line 6:      49: getstatic     #7                  // Field java/lang/System.out:Ljava/io/PrintStream;
line 6:      52: ldc           #19                 // String default
line 6:      54: invokevirtual #21                 // Method java/io/PrintStream.println:(Ljava/lang/String;)V
line 8:      57: return

}