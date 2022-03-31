const std = @import("std");
const mem = std.mem;
const assert = std.debug.assert;

// TODO (Matteo): Unit tests

/// 'Buffer' of flat memory
pub const MemBuffer = Buffer(u8);

/// Generic linear buffer of items, represented as {pointer, length, capacity}.
/// For ease of use, pointer and length are exposed with the 'items' slice.
/// Offers both a static and dynamic memory interface (the latter requires passing
/// an allocator, and is it user's responibility to be consistent).
/// Push/pop operations are constant time (amortized in case of dynamic growth).
pub fn Buffer(comptime T: type) type {
    return struct {
        items: []T = &[_]T{},
        capacity: usize,

        pub const Error = mem.Allocator.Error || error{BufferFull};

        const Self = @This();

        //=== Non-allocating API ===//

        pub fn fromMemory(memory: []T) Self {
            var self = Self{ .items = memory, .capacity = memory.len };
            self.items.len = 0;
            return self;
        }

        pub fn clear(self: *Self) void {
            self.items.len = 0;
        }

        pub fn resize(self: *Self, size: usize) Error!void {
            if (size > self.capacity) return Error.BufferFull;
            self.len = size;
        }

        pub fn extend(self: *Self, amount: usize) Error!void {
            try resize(self, self.items.len + amount);
        }

        pub fn shrink(self: *Self, amount: usize) void {
            try resize(self, self.items.len - amount);
        }

        pub fn push(self: *Self, item: T) Error!void {
            const at = self.items.len;
            try self.resize(at + 1);
            self.items[at] = item;
        }

        pub fn pop(self: *Self) T {
            const at = self.items.len - 1;
            const item = self.items[at];
            self.len = at;
            return item;
        }

        pub fn stableRemove(self: *Self, at: usize) T {
            const len = self.items.len;
            if (at == len) return self.pop();

            const item = self.items[at];

            mem.copy(T, self.items[at .. len - 1], self.items[at + 1 .. len]);
            self.items.len = len - 1;

            return item;
        }

        pub fn swapRemove(self: *Self, at: usize) T {
            const item = self.items[at];
            const last = self.items.len - 1;
            if (last != at) self.items[at] = self.items[last];
            self.items.len = last;
            return item;
        }

        //=== Allocating API ===//

        pub fn allocate(capacity: usize, allocator: mem.Allocator) Error!Self {
            const memory = try allocator.alloc(T, capacity);
            var self = Self{ .items = memory, .capacity = memory.len };
            self.items.len = 0;
            return self;
        }

        pub fn free(self: *Self, allocator: mem.Allocator) void {
            self.clear();
            allocator.free(self.fullSlice());
        }

        pub fn ensureCapacity(
            self: *Self,
            req: usize,
            allocator: mem.Allocator,
        ) Error!void {
            if (req > self.capacity) {
                var new_cap = self.capacity * 2;
                if (new_cap < req) new_cap = req;
                const old_mem = self.fullSlice();
                const new_mem = try allocator.reallocAtLeast(old_mem, new_cap);
                self.items.ptr = new_mem.ptr;
                self.capacity = new_mem.len;
            }
        }

        pub fn resizeAlloc(
            self: *Self,
            size: usize,
            allocator: mem.Allocator,
        ) Error!void {
            try self.ensureCapacity(size, allocator);
            try self.resize(size);
        }

        pub fn extendAlloc(
            self: *Self,
            amount: usize,
            allocator: mem.Allocator,
        ) Error!void {
            try self.resizeAlloc(self.items.len + amount, allocator);
        }

        pub fn pushAlloc(
            self: *Self,
            item: T,
            allocator: mem.Allocator,
        ) Error!void {
            const at = self.items.len;
            try self.resizeAlloc(at + 1, allocator);
            self.items[at] = item;
        }

        //=== Internals ===//

        fn fullSlice(self: Self) []T {
            return self.items.ptr[0..self.capacity];
        }
    };
}

/// Generic circular buffer
/// Offers basic FIFO operations (pushFront/popBack) but can be also used as a 
/// deque (pushFront/popBack).
/// Offers optional allocating functions for dynamically growing the buffer (it 
/// is user's responibility to be consistent with the passed allocator).
pub fn RingBuffer(comptime T: type) type {

    // NOTE (Matteo): Here I'm using a "virtual stream" technique, with the
    // front and back indices monotonically increasing and the 'modulo size' operation
    // used only for item access.
    // This relies on the size being a power of 2 and unsigned wrapping on overflow
    // (which is the standard behavior in C while Zig uses explicit operators).
    // The deque case is a bit odd here since the indices would be decremented in
    // that case, possibly resulting in a "strange" front-back pair.

    // TODO (Matteo): Provide common ring buffer "streaming" operations
    // (i.e. read/write of slices in FIFO order)

    return struct {
        items: []T = &[_]T{},
        back: usize = 0,
        front: usize = 0,
        mask: usize = 0,

        const Self = @This();

        pub const Error = Buffer(T).Error || error{SizeNotPowerOfTwo};

        //=== Non-allocating API ===//

        pub fn fromMemory(memory: []T) Error!Self {
            if (!std.math.isPowerOfTwo(memory.len)) return Error.SizeNotPowerOfTwo;
            return Self{ .items = memory, .mask = memory.len - 1 };
        }

        pub inline fn count(self: Self) usize {
            return self.back -% self.front;
        }

        pub inline fn peekFront(self: Self) ?T {
            return if (self.count() == 0)
                null
            else
                self.items[self.front & self.mask];
        }

        pub inline fn peekBack(self: Self) ?T {
            return if (self.count() == 0)
                null
            else
                self.items[(self.back -% 1) & self.mask];
        }

        pub fn clear(self: *Self) void {
            self.front = 0;
            self.back = 0;
        }

        // FIFO behavior

        pub fn pushBack(self: *Self, item: T) Error!void {
            if (self.count() == self.items.len) return Error.BufferFull;
            self.items[self.back & self.mask] = item;
            self.back = self.back +% 1;
        }

        pub fn popFront(self: *Self) ?T {
            if (self.count() == 0) return null;
            const item = self.items[self.front & self.mask];
            self.front = self.front +% 1;
            return item;
        }

        // Additional deque behavior

        pub fn pushFront(self: *Self, item: T) Error!void {
            if (self.count() == self.items.len) return Error.BufferFull;

            self.front = self.front -% 1;
            self.items[self.front & self.mask] = item;
        }

        pub fn popBack(self: *Self) ?T {
            if (self.count() == self.items.len) return Error.BufferFull;

            self.back = self.back -% 1;
            return self.items[self.back & self.mask];
        }

        //=== Allocating API ===//

        pub fn allocate(allocator: mem.Allocator, capacity: usize) Error!Self {
            const memory = try allocator.alloc(T, capacity);
            errdefer allocator.free(memory);
            return fromMemory(memory);
        }

        pub fn free(self: *Self, allocator: mem.Allocator) void {
            self.clear();
            allocator.free(self.items);
        }

        pub fn ensureCapacity(self: *Self, allocator: mem.Allocator, req: usize) Error!void {
            if (!std.math.isPowerOfTwo(req)) return Error.SizeNotPowerOfTwo;

            if (req > self.items.len) {
                var new_cap = self.items.len * 2;
                if (new_cap < req) new_cap = req;

                assert(std.math.isPowerOfTwo(new_cap));

                const old_mem = self.fullSlice();
                const new_mem = try allocator.realloc(old_mem, new_cap);

                assert(new_mem.len == new_cap);

                self.items = new_mem;
                self.mask = self.items.len - 1;
            }
        }

        pub fn pushBackAlloc(
            self: *Self,
            item: T,
            allocator: mem.Allocator,
        ) Error!void {
            try self.ensureCapacity(self.count() + 1, allocator);
            try self.pushBack(item);
        }

        pub fn pushFrontAlloc(
            self: *Self,
            item: T,
            allocator: mem.Allocator,
        ) Error!void {
            try self.ensureCapacity(self.count() + 1, allocator);
            try self.pushFront(item);
        }
    };
}

//=== Testing ===//

const expect = std.testing.expect;

test "RingBuffer - Static FIFO" {
    var buf = [_]u32{0} ** 16;
    var ring = try RingBuffer(u32).fromMemory(&buf);

    try expect(ring.count() == 0);
    try expect(ring.peekFront() == null);
    try expect(ring.peekBack() == null);

    try ring.pushBack(0);
    try ring.pushBack(1);
    try ring.pushBack(2);

    try expect(ring.count() == 3);
    try expect(ring.peekFront().? == 0);
    try expect(ring.peekBack().? == 2);

    try expect(ring.popFront().? == 0);
    try expect(ring.popFront().? == 1);
    try expect(ring.popFront().? == 2);

    try expect(ring.count() == 0);
    try expect(ring.peekFront() == null);
    try expect(ring.peekBack() == null);
}

test "RingBuffer - Static deque" {
    var buf = [_]u32{0} ** 16;
    var ring = try RingBuffer(u32).fromMemory(&buf);

    try expect(ring.count() == 0);
    try expect(ring.peekFront() == null);
    try expect(ring.peekBack() == null);

    try ring.pushBack(2);
    try ring.pushFront(1);
    try ring.pushBack(3);
    try ring.pushFront(0);

    try expect(ring.count() == 4);
    try expect(ring.peekFront().? == 0);
    try expect(ring.peekBack().? == 3);

    try expect(ring.popFront().? == 0);
    try expect(ring.popFront().? == 1);
    try expect(ring.popFront().? == 2);

    try expect(ring.count() == 1);
    try expect(ring.peekFront().? == 3);
    try expect(ring.peekBack().? == 3);
}
