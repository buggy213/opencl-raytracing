#pragma once

#if defined _WIN32
    #define DLL_IMPORT __declspec(dllimport)
    #define DLL_EXPORT __declspec(dllexport)
    #define DLL_LOCAL
#else
    #define DLL_IMPORT __attribute__((visibility("default")))
    #define DLL_EXPORT __attribute__((visibility("default")))
    #define DLL_LOCAL __attribute__((visibility("hidden")))
#endif


#ifdef SHARED_LIBARY
    #ifdef BUILDING_SHARED_LIBRARY
        // we are being built as a shared library
        #define RT_API DLL_EXPORT
    #else
        // we are using this symbol from a shared library
        #define RT_API DLL_IMPORT
    #endif
#else
    // we are being built or using this header as part of a static library, so visibility is unimportant
    #define RT_API
#endif