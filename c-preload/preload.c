#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>

#ifndef O_CREAT
#define O_CREAT 0100
#endif

static int allowed_env(const char* pathname, const char* envvar) {
    char* dirpathbuf = strdup(pathname);
    char* dirpath = dirname(dirpathbuf);

    char resolvedBuf[PATH_MAX];
    char* resolved = realpath(dirpath, resolvedBuf);
    free(dirpathbuf);
    if (resolved == NULL) {
        return 0;
    }

    const char* okpath = getenv(envvar);
    if (okpath == NULL) {
       errno = EINVAL;
       return 0;
    }

    while (*okpath) {
        const char* end = strchrnul(okpath, ':');
        if (strncmp(okpath, resolved, end - okpath) == 0) return 1;
        okpath = end;
        while (*okpath == ':') ++okpath;
    }

    fprintf(stderr, "Denying access to \"%s\"\n", pathname);
    errno = EACCES;
    return 0;
}

static int allowed(const char* pathname, int flags) {
    if (flags & O_CREAT)
        return allowed_env(pathname, "ALLOWED_FOR_CREATE");
    else
        return allowed_env(pathname, "ALLOWED_FOR_READ");
}

int open(const char *pathname, int flags, mode_t mode) {
    static int (*real_open)(const char*, int, mode_t) = NULL;
    if (!real_open) real_open = dlsym(RTLD_NEXT, "open");

    if (!allowed(pathname, flags)) {
        return -1;
    }

    return real_open(pathname, flags, mode);
}

int creat(const char *pathname, mode_t mode) {
    static int (*real_creat)(const char*, mode_t) = NULL;
    if (!real_creat) real_creat = dlsym(RTLD_NEXT, "creat");

    if (!allowed(pathname, O_CREAT)) {
        return -1;
    }

    return real_creat(pathname, mode);
}

FILE* fopen(const char* name, const char* mode) {
    static FILE* (*real_fopen)(const char*, const char*) = NULL;
    if (!real_fopen) real_fopen = dlsym(RTLD_NEXT, "fopen");

    if (!allowed(name, (mode[0] == 'r') ? 0 : O_CREAT)) {
        return NULL;
    }

    return real_fopen(name, mode);
}
