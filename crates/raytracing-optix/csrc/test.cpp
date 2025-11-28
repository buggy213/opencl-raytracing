#include "lib.h"

#include <cstdio>

void host_code();

struct Scene;

extern "C" {
    RT_API void render(Scene* scene) {

    }

    RT_API void test() {
        printf("hi from cpp 2 asdf asdf\n");
        host_code();
    }
}