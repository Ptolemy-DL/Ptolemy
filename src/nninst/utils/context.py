import threading


class ThreadLocalStack(threading.local):
    """A thread-local stack."""

    def __init__(self):
        super(ThreadLocalStack, self).__init__()
        self.stack = []
        # self.lock = threading.Lock()

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        self.stack.pop()

    def __iter__(self):
        return reversed(self.stack).__iter__()

    def swap_stack(self, new_stack):
        old_stack = self.stack
        self.stack = new_stack
        return old_stack


class Stack:
    def __init__(self):
        self.stack = []
        self.lock = threading.Lock()

    def push(self, item):
        with self.lock:
            self.stack.append(item)

    def pop(self):
        with self.lock:
            self.stack.pop()

    def __iter__(self):
        with self.lock:
            return reversed(self.stack).__iter__()

    def swap_stack(self, new_stack):
        with self.lock:
            old_stack = self.stack
            self.stack = new_stack
            return old_stack


class Context(object):
    def __init__(self):
        # self._configs = Stack()
        self._configs = ThreadLocalStack()

    @property
    def configs(self):
        return self._configs


_context: Context = None
_context_lock = threading.Lock()


def _initialize_context():
    global _context
    with _context_lock:
        if _context is None:
            _context = Context()


def context():
    """Returns a singleton context object."""
    if _context is None:
        _initialize_context()
    return _context
