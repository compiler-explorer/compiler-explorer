#include <cstdio>
#include <cstring>

double testFunction(double* input, double length) {
  double sum = 0;
  for (int i = 0; i < length; ++i) {
    sum += input[i];
  }
  return sum;
}

int fibo(int n) 
{ 
    if (n <= 1) 
        return n; 
    return fibo(n-1) + fibo(n-2); 
} 

int fizz_buzz()
{
    int i;
    for(i=1; i<=100; ++i)
    {
        if (i % 3 == 0)
            printf("Fizz");
        if (i % 5 == 0)
            printf("Buzz");
        if ((i % 3 != 0) && (i % 5 != 0))
            printf("number=%d", i);
        printf("\n");
    }

    return 0;
}

// Function to print even numbers 
void printEvenNumbers(int N) 
{
    for (int i = 1; i <= 2 * N; i++) { 
  
        // Numbers that are divisible by 2 
        if (i % 2 == 0) 
            printf("%d", i); 
    } 
} 
  
// Function to print odd numbers 
void printOddNumbers(int N) 
{
    for (int i = 1; i <= 2 * N; i++) { 
  
        // Numbers that are not divisible by 2 
        if (i % 2 != 0) 
            printf("%d", i);
    } 
}

void tokenizeString()
{ 
    char str[] = "Geeks-for-Geeks"; 
  
    // Returns first token  
    char *token = strtok(str, "-"); 
    
    // Keep printing tokens while one of the 
    // delimiters present in str[]. 
    while (token != NULL) 
    { 
        printf("%s\n", token); 
        token = strtok(NULL, "-"); 
    }
} 

int main () 
{ 
    int n = fibo(10);
    fizz_buzz();
    printOddNumbers(n);
    double a[] = {1, 2, 3, 4, 5};
    double sum = testFunction(a, 5);
    tokenizeString();
    return 0; 
}
