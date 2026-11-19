#include <execinfo.h>
#include <signal.h>
#include <unistd.h>

static void dump_stack(int number) {
    void *frames[48];
    const int count = backtrace(frames, 48);
    static const char prefix[] = "shared_fascia_signal_backtrace\n";
    (void)write(STDERR_FILENO, prefix, sizeof(prefix) - 1);
    backtrace_symbols_fd(frames, count, STDERR_FILENO);
    _exit(128 + number);
}

__attribute__((constructor)) static void install_trace(void) {
    void *warm[1];
    (void)backtrace(warm, 1);
    struct sigaction action = {0};
    action.sa_handler = dump_stack;
    sigemptyset(&action.sa_mask);
    sigaction(SIGSEGV, &action, 0);
    sigaction(SIGBUS, &action, 0);
}
