#include <stdio.h>

void host_code();

extern "C" {
    __declspec(dllexport) void test() {
        printf("hi from cpp 2 asdf asdf\n");
        host_code();
    }
}