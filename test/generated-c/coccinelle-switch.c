#include <stdio.h>
#define LIMIT 10

/* Classify a number */
int classify(int x) {
	switch (x) {
	case 1:
		return 20;
	default:
		return 0;
	}
}

int main(void) {
	// print it
	printf("%d\n", classify(2));
	return 0;
}

