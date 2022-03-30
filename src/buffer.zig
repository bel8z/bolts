const std = @import("std");
const mem = std.mem;

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

        pub fn allocate(allocator: mem.Allocator, capacity: usize) Error!Self {
            const memory = try allocator.alloc(T, capacity);
            var self = Self{ .items = memory, .capacity = memory.len };
            self.items.len = 0;
            return self;
        }

        pub fn free(self: *Self, allocator: mem.Allocator) void {
            self.clear();
            allocator.free(self.fullSlice());
        }

        pub fn ensureCapacity(self: *Self, allocator: mem.Allocator, req: usize) Error!void {
            if (req > self.capacity) {
                var new_cap = self.capacity * 2;
                if (new_cap < req) new_cap = req;
                const old_mem = self.fullSlice();
                const new_mem = try allocator.reallocAtLeast(old_mem, new_cap);
                self.items.ptr = new_mem.ptr;
                self.capacity = new_mem.len;
            }
        }

        pub fn resizeAlloc(self: *Self, allocator: mem.Allocator, size: usize) Error!void {
            try self.ensureCapacity(allocator, size);
            try self.resize(size);
        }

        pub fn extendAlloc(self: *Self, allocator: mem.Allocator, amount: usize) Error!void {
            try resizeAlloc(self, allocator, self.items.len + amount);
        }

        pub fn pushAlloc(self: *Self, allocator: mem.Allocator, item: T) Error!void {
            const at = self.items.len;
            try self.resizeAlloc(allocator, at + 1);
            self.items[at] = item;
        }

        //=== Internals ===//

        fn fullSlice(self: Self) []T {
            return self.items.ptr[0..self.capacity];
        }
    };
}

pub fn RingBuffer(comptime T: type) type {
    return struct {
        items: []T = &[_]T{},
        head: usize,
        tail: usize,
    };
}
