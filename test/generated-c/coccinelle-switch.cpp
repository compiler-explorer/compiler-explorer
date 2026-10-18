#include <cstdio>
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

class Printer {
public:
	void print(int x) { std::printf("%d\n", x); }
};

int main() {
	// print it
	Printer().print(classify(2));
	return 0;
}

