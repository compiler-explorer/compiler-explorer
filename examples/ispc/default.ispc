// Type your code here, or load an example.
varying int square_even(varying int num) {
    if (num % 2 == 0) {
        return num * num;
    } else {
        return num;
    }
}

// Uncomment this if you would like to execute the code
/*
extern "C" uniform int main() {
    print("Hello, World!\n");
    print("programCount = %\n", programCount);
    print("square_even(%) = %\n", programIndex, square_even(programIndex));
    return 0;
}
*/
