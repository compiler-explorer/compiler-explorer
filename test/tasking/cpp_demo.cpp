#include <iostream>
using namespace std;
void printfhello()
{
    printf("helloworld!!");
}

int main( )
{
    int j = 0;
    for(int i = 0; i<10;i++){
       j += 100;
    }

    printfhello();
   
   return j*4;
}