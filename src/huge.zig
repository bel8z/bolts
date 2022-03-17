//! This module provides:
//! * Bump: a simple, bump allocator over a (possibly) huge virtual memory block.
//!   Only the last allocation can be freed, but LIFO-ordered frees are not ensured; 
//!   instead functionality for saving and restoring state (BumpState) are provided 
//!   for temporary allocations.
//! * Stack: a stack allocator over a (possibly) huge virtual memory block.
//!   LIFO ordering is ensured for frees, therefore there is no need to save/restore
//!   state for temporary allocation. However, this causes a slightly higher memory
//!   footprint in order for padding information to be stored.
//!
//! Virtual memory address-space is reserved ahead of time, but is committed upon usage.
//! Memory is decommitted on frees for safe builds (Debug/ReleaseSafe); this triggers 
//! segfaults on use-after-free at the cost of more system calls. 
//! In unsafe builds memory is never decommitted, so heavy temporary memory usage 
//! doesn't cause a lot of syscalls, but use-after-free cannot be detected 
//! deterministically (memory is zeroed anyways, so segfaults can still happen). 

//==== Imports ====//

const std = @import("std");
const math = std.math;
const mem = std.mem;
const os = std.os;
const assert = std.debug.assert;

//==== Common exports ====//

/// Re-exported allocator errors
pub const Error = mem.Allocator.Error;

/// Size of virtual memory pages
pub const page_size = mem.page_size;

//==== Bump ====//

/// State of the 'Bump' allocator, useful for temporary, scoped allocations.
pub const BumpState = struct {
    bump: *Bump,
    index: usize,
    allocated: usize,

    const Self = @This();

    pub fn allocator(self: *Self) mem.Allocator {
        return self.bump.allocator();
    }

    pub fn restore(self: *Self) !void {
        return self.bump.restoreState(self);
    }
};

/// Bump allocator over a (possibly) huge virtual memory reservation.
pub const Bump = struct {
    buffer: []u8,
    committed: usize = 0,
    allocated: usize = 0,
    save_stack: usize = 0,

    const Self = @This();
    const win32 = os.windows;

    comptime {
        assert(@import("builtin").os.tag == .windows);
    }

    /// Construct a huge bump allocator with the given capacity.
    /// Capacity is rounded up to a multiple of 'page_size'.
    pub fn init(capacity: usize) Error!Self {
        const aligned_len = mem.alignForward(capacity, page_size);
        const addr = win32.VirtualAlloc(
            null,
            capacity,
            win32.MEM_RESERVE,
            win32.PAGE_NOACCESS,
        ) catch return Error.OutOfMemory;

        return Self{ .buffer = @ptrCast([*]u8, addr)[0..aligned_len] };
    }

    /// 
    pub fn deinit(self: *Self) void {
        win32.VirtualFree(self.buffer.ptr, 0, win32.MEM_RELEASE);
        // Invalidate allocation
        self.buffer = self.buffer[0..0];
        self.committed = 0;
        self.allocated = 0;
    }

    /// Provide the type-erased allocator implementation
    pub fn allocator(self: *Self) mem.Allocator {
        return mem.Allocator.init(self, alloc, resize, free);
    }

    pub fn saveState(self: *Self) BumpState {
        self.save_stack += 1;
        return BumpState{
            .bump = self,
            .index = self.save_stack,
            .allocated = self.allocated,
        };
    }

    pub fn restoreState(self: *Self, state: *BumpState) !void {
        if (self.save_stack != state.index) return error.InvalidState;
        if (self.allocated < state.allocated) return error.InvalidState;

        self.popTo(state.allocated);

        // TODO (Matteo): Better approach?
        // Invalidate state
        state.index = 0;
    }

    /// Reset all allocations.
    /// BEWARE OF DANGLING POINTERS!
    pub inline fn reset(self: *Self) void {
        self.popTo(0);
    }

    pub fn alloc(
        self: *Self,
        len: usize,
        ptr_align: u29,
        len_align: u29,
        return_address: usize,
    ) Error![]u8 {
        _ = len_align;
        _ = return_address;

        const offset = self.nextPadding(ptr_align) orelse return Error.OutOfMemory;
        return self.push(len, offset);
    }

    pub fn resize(
        self: *Self,
        buf: []u8,
        buf_align: u29,
        new_len: usize,
        len_align: u29,
        return_address: usize,
    ) ?usize {
        _ = buf_align;
        _ = len_align;
        _ = return_address;

        // Handle shrink request
        if (new_len <= buf.len) {
            if (self.isLastAlloc(buf)) self.pop(buf.len - new_len);
            return new_len;
        }

        // Handle expansion request
        if (!self.isLastAlloc(buf)) return null;

        const room = new_len - buf.len;
        if (self.allocated + room > self.buffer.len) return null;

        self.allocated += room;
        self.commitAllocation();

        return new_len;
    }

    pub fn free(
        self: *Self,
        buf: []u8,
        buf_align: u29,
        return_address: usize,
    ) void {
        _ = buf_align;
        _ = return_address;

        if (self.isLastAlloc(buf)) {
            self.pop(buf.len);
        }
    }

    inline fn isLastAlloc(self: Self, buf: []u8) bool {
        assert(@ptrToInt(buf.ptr) >= @ptrToInt(self.buffer.ptr));

        return (buf.ptr + buf.len) == (self.buffer.ptr + self.allocated);
    }

    inline fn nextPadding(self: Self, ptr_align: u29) ?usize {
        return mem.alignPointerOffset(self.buffer.ptr + self.allocated, ptr_align);
    }

    inline fn push(self: *Self, len: usize, padding: usize) Error![]u8 {
        // Provide allocation
        const buffer_start = self.allocated + padding;
        const buffer_end = buffer_start + len;
        if (buffer_end > self.buffer.len) return Error.OutOfMemory;

        self.allocated = buffer_end;
        self.commitAllocation();

        return self.buffer[buffer_start..buffer_end];
    }

    inline fn pop(self: *Self, size: usize) void {
        assert(size <= self.allocated);
        self.popTo(self.allocated - size);
    }

    fn popTo(self: *Self, allocated: usize) void {
        assert(allocated <= self.allocated);

        self.allocated = allocated;

        // NOTE (Matteo): Decommit starting from the first fully unused page
        const watermark = mem.alignForward(self.allocated, page_size);

        if (std.debug.runtime_safety and watermark < self.committed) {
            win32.VirtualFree(
                self.buffer.ptr + watermark,
                self.committed - watermark,
                win32.MEM_DECOMMIT,
            );

            self.committed = watermark;
        }

        // NOTE (Matteo): Clear excess buffer space for consistency
        mem.set(u8, self.buffer[self.allocated..self.committed], 0);
    }

    /// Ensure the allocated bytes are properly committed
    fn commitAllocation(self: *Self) void {
        if (self.allocated > self.committed) {
            // NOTE (Matteo): Align memory commits to page boundaries
            const alloc_pos = @ptrToInt(self.buffer.ptr + self.allocated);
            const curr_pos = @ptrToInt(self.buffer.ptr + self.committed);
            const next_pos = mem.alignForward(alloc_pos, page_size);

            const max_commit_size = self.buffer.len - self.committed;
            const commit_size = math.min(next_pos - curr_pos, max_commit_size);

            _ = win32.VirtualAlloc(
                @intToPtr(*u8, curr_pos),
                commit_size,
                win32.MEM_COMMIT,
                win32.PAGE_READWRITE,
            ) catch unreachable;

            self.committed += commit_size;
        }
    }
};

//==== Stack ====//

/// Stack allocator over a (possibly) huge virtual memory reservation.
pub const Stack = struct {
    bump: Bump,

    const Self = @This();

    const max_align = 2 << 7;
    const max_padding = math.maxInt(u8);

    /// Construct a huge stack allocator with the given capacity.
    /// Capacity is rounded up to a multiple of 'page_size'.
    pub fn init(capacity: usize) Error!Self {
        return Self{ .bump = try Bump.init(capacity) };
    }

    pub fn deinit(self: *Self) void {
        self.bump.deinit();
    }

    /// Provide the type-erased allocator implementation
    pub fn allocator(self: *Self) mem.Allocator {
        return mem.Allocator.init(self, alloc, resize, free);
    }

    /// Reset all allocations.
    /// BEWARE OF DANGLING POINTERS!
    pub inline fn reset(self: *Self) void {
        self.bump.popTo(0);
    }

    pub fn alloc(
        self: *Self,
        len: usize,
        ptr_align: u29,
        len_align: u29,
        return_address: usize,
    ) Error![]u8 {
        _ = len_align;
        _ = return_address;

        if (ptr_align > max_align) return Error.OutOfMemory;

        // Compute required padding
        var padding = self.bump.nextPadding(ptr_align) orelse return Error.OutOfMemory;

        if (padding == 0) {
            // Make room for the padding byte
            padding += ptr_align * (1 + 1 / ptr_align);
        }

        assert(padding <= max_padding);

        // Provide allocation
        var buf = try self.bump.push(len, padding);

        // Write padding byte before the provided allocation
        var pad_ptr = buf.ptr - 1;
        pad_ptr[0] = @intCast(u8, padding);

        return buf;
    }

    pub fn resize(
        self: *Self,
        buf: []u8,
        buf_align: u29,
        new_len: usize,
        len_align: u29,
        return_address: usize,
    ) ?usize {
        return self.bump.resize(buf, buf_align, new_len, len_align, return_address);
    }

    pub fn free(
        self: *Self,
        buf: []u8,
        buf_align: u29,
        return_address: usize,
    ) void {
        _ = buf_align;
        _ = return_address;

        if (self.bump.isLastAlloc(buf)) {
            // Shrink allocation by buffer size and padding
            const pad_ptr = buf.ptr - 1;
            const pad = pad_ptr[0];
            self.bump.pop(buf.len + pad);
        }
    }
};

//==== Pool ====//

// TODO (Matteo): Implement

//==== Free list ====//

// TODO (Matteo): Test it!

/// Implements a free list allocator over a (possibly) huge 'Stack'.
///
/// Uses a doubly-linked list to keep free blocks sorted by address (which allows 
/// a simple auto-defragmentation on frees), and multiple singly-linked list of blocks
/// divided by size (in order to reduce fragmentation in the first place). 
/// 
/// A minimum allocation size is enforced only for storing 'Node' metadata in case
/// of free blocks, but otherwise no limitation is applied. This isn't ideal because
/// it can cause fragmentation, but actually I wasn't able to find a method that
/// plays well with the standard 'Allocator' 'len_align' requirement.
pub const FreeList = struct {
    bump: Bump,
    large: Node,
    buckets: [bucket_count]Node = undefined,

    const Self = @This();

    const node_size = @sizeOf(Node);
    const node_align = @alignOf(Node);
    const bucket_count = math.log2(page_size) - math.log2(node_size) + 1;

    comptime {
        assert(math.isPowerOfTwo(node_size));
        assert(math.isPowerOfTwo(node_align));
        assert(page_size & (node_size) - 1 == 0);
    }

    pub fn init(self: *Self, capacity: usize) !void {
        // NOTE (Matteo): Initialization expects a pointer to a previously allocated
        // object because a permanent pointer to the sentinel nodes is required.

        self.bump = try Bump.init(capacity);
        self.large.next = &self.large;
        self.large.prev = self.large.next;

        var bucket: usize = 0;
        while (bucket < bucket_count) : (bucket += 1) {
            var sentinel = &self.buckets[bucket];
            self.buckets[bucket].next = sentinel;
            self.buckets[bucket].prev = sentinel;
        }
    }

    pub fn deinit(self: *Self) void {
        for (self.buckets) |_, i| {
            self.buckets[i].remove();
        }
        self.large.remove();
        self.bump.deinit();
    }

    /// Provide the type-erased allocator implementation
    pub fn allocator(self: *Self) mem.Allocator {
        return mem.Allocator.init(self, alloc, resize, free);
    }

    pub fn alloc(
        self: *Self,
        len: usize,
        ptr_align: u29,
        len_align: u29,
        return_address: usize,
    ) Error![]u8 {
        _ = return_address;
        _ = len_align;

        // Check that the required alignment can be satisfied
        assert(math.isPowerOfTwo(ptr_align));
        if (ptr_align < node_align and node_align & (ptr_align - 1) != 0) return error.OutOfMemory;

        // Allocate in multiple of 'node_size'
        const padding = mem.alignForward(node_align, ptr_align) - node_align;
        const block_size = mem.alignForward(padding + len, node_size);

        // TODO (Matteo): Handle larger allocations
        if (block_size > page_size) return error.OutOfMemory;

        var bucket_index = getSizeBucket(block_size);

        var buf: []u8 = undefined;
        var next = self.buckets[bucket_index].next;

        if (next != &self.buckets[bucket_index]) {
            next.remove();
            // Clear memory for consistency (the memory provided by the stack is always zeroed)
            var ptr = @ptrCast([*]u8, next);
            mem.set(u8, ptr[padding .. padding + len], 0);
            buf = ptr[0..block_size];
        } else {
            var storage_padding = self.bump.nextPadding(node_align) orelse return Error.OutOfMemory;
            assert(storage_padding == 0 or self.bump.allocated == 0);

            buf = try self.bump.push(block_size, storage_padding);
        }

        // Write a magic value in the padding portion for debugging
        mem.set(u8, buf[0..padding], 0xDD);

        return buf[padding .. padding + len];
    }

    pub fn resize(
        self: *Self,
        buf: []u8,
        buf_align: u29,
        new_len: usize,
        len_align: u29,
        return_address: usize,
    ) ?usize {
        // Allocate in multiple of 'node_size'
        const padding = mem.alignForward(node_align, buf_align) - node_align;
        const old_size = mem.alignForward(padding + buf.len, node_size);
        const new_size = mem.alignForward(padding + new_len, node_size);

        // TODO (Matteo): Handle larger allocations
        if (new_size > page_size) return null;

        // Retrieve the full allocated block
        const block_addr = mem.alignBackward(@ptrToInt(buf.ptr), node_align);
        var block: []u8 = undefined;
        block.ptr = @intToPtr([*]u8, block_addr);
        block.len += buf.len + @ptrToInt(buf.ptr) - block_addr;

        assert(@ptrToInt(buf.ptr) - block_addr == padding);
        assert(block.len == old_size);

        if (old_size - padding > new_size) {
            // Reallocate in place
            return new_len;
        } else if (self.bump.isLastAlloc(block)) {
            // Try to reallocate on top of the stack
            if (self.bump.resize(block, node_align, new_size, len_align, return_address)) |sz| {
                assert(sz == new_size);
                return new_len;
            }
        }

        return null;
    }

    pub fn free(
        self: *Self,
        buf: []u8,
        buf_align: u29,
        return_address: usize,
    ) void {
        _ = buf_align;
        _ = return_address;

        // Allocate in multiple of 'node_size'
        const padding = mem.alignForward(node_align, buf_align) - node_align;

        // Retrieve the full allocated block
        const block_addr = mem.alignBackward(@ptrToInt(buf.ptr), node_align);

        var block: []u8 = undefined;
        block.ptr = @intToPtr([*]u8, block_addr);
        block.len = mem.alignForward(padding + buf.len, node_size);

        assert(@ptrToInt(buf.ptr) - block_addr == padding);

        if (self.bump.isLastAlloc(block)) {
            // Allocation can return to the stack
            self.bump.pop(block.len);
        } else {
            // Store current block in the free list, keeping it ordered by address
            var curr = @ptrCast(*Node, @alignCast(node_align, block.ptr));
            self.insertBySize(curr, block.len);
        }
    }

    fn insertBySize(
        self: *Self,
        node: *Node,
        size: usize,
    ) void {
        var bucket_index = getSizeBucket(size);

        var sentinel = &self.buckets[bucket_index];
        var next = sentinel.next;

        while (next != sentinel) : (next = next.next) {
            if (@ptrToInt(next) > @ptrToInt(node)) break;
        }

        node.insert(next.prev, next);

        assert(sentinel.next != sentinel);
        assert(sentinel.prev != sentinel);
    }

    fn getSizeBucket(size: usize) usize {
        // const size_class_hint = math.ceilPowerOfTwoAssert(usize, size);
        // var bucket_index = math.log2(size_class_hint);

        var bucket_index = math.log2(size / node_size);

        if (bucket_index < 0) {
            bucket_index = 0;
        } else if (bucket_index > bucket_count) {
            bucket_index = bucket_count;
        }

        return bucket_index;
    }

    const Node = struct {
        prev: *Node,
        next: *Node,

        inline fn remove(self: *Node) void {
            // Rewire prev and next links
            self.prev.next = self.next;
            self.next.prev = self.prev;
            // Make the node a circular list of itself
            self.next = self;
            self.prev = self;
        }

        inline fn insert(self: *Node, prev: *Node, next: *Node) void {
            // Assign links
            self.prev = prev;
            self.next = next;
            // Rewire prev and next links
            self.prev.next = self;
            self.next.prev = self;
        }

        inline fn replace(self: *Node, replacement: *Node) void {
            // Assign links
            replacement.prev = self.prev;
            replacement.next = self.next;
            // Rewire prev and next links
            replacement.prev.next = self;
            replacement.next.prev = self;
            // Make the node a circular list of itself
            self.next = self;
            self.prev = self;
        }

        fn tryCoalesce(self: *Node, sentinel: *Node) *Node {
            var result = self;
            const prev = result.prev;
            const next = result.next;

            if (prev != sentinel and (@ptrToInt(prev) + prev.size) == @ptrToInt(result)) {
                prev.size += result.size;
                result.remove();
                result = prev;
            }

            if (next != sentinel and (@ptrToInt(result) + result.size) == @ptrToInt(next)) {
                result.size += next.size;
                next.remove();
            }

            return result;
        }
    };
};

//==== Testing ====//

test "Huge - Alignment" {
    const node_size = 32;
    std.debug.print("\n", .{});
    std.debug.print("Node size = {}\n", .{FreeList.node_size});
    std.debug.print("Node alignment = {}\n", .{FreeList.node_align});
    std.debug.print(
        " mem.alignBackward({}, {}) = {}\n",
        .{ 0, FreeList.node_align, mem.alignBackward(0, FreeList.node_align) },
    );
    std.debug.print(
        " mem.alignBackward({}, {}) = {}\n",
        .{
            FreeList.node_align,
            FreeList.node_align,
            mem.alignBackward(FreeList.node_align, FreeList.node_align),
        },
    );
    std.debug.print(
        " mem.alignBackward({}, {}) = {}\n",
        .{ 32, FreeList.node_align, mem.alignBackward(32, FreeList.node_align) },
    );
    std.debug.print(
        " mem.alignForward(24, node_size) = {}\n",
        .{mem.alignForward(24, node_size)},
    );
    std.debug.print(
        " mem.alignForward(32, node_size) = {}\n",
        .{mem.alignForward(32, node_size)},
    );
    std.debug.print(
        " mem.alignForward(127, node_size) = {}\n",
        .{mem.alignForward(127, node_size)},
    );
    std.debug.print(
        " mem.alignForward(128, node_size) = {}\n",
        .{mem.alignForward(128, node_size)},
    );
    std.debug.print(
        " mem.alignForward(4000, node_size) = {}\n",
        .{mem.alignForward(4000, node_size)},
    );
    std.debug.print(
        " mem.alignForward(4096, node_size) = {}\n",
        .{mem.alignForward(4096, node_size)},
    );
    std.debug.print(
        " mem.alignForward(5000, node_size) = {}\n",
        .{mem.alignForward(5000, node_size)},
    );
}

test "Huge - Bump state" {
    var bump = try Bump.init(1024 * 1024);
    var alloc = bump.allocator();
    defer bump.deinit();

    _ = try alloc.alloc(u8, 794);

    try std.testing.expect(bump.allocated == 794);

    {
        var tmp = bump.saveState();
        defer tmp.restore() catch unreachable;

        _ = try alloc.alloc(u8, 210);

        try std.testing.expect(bump.allocated == 794 + 210);
    }

    try std.testing.expect(bump.allocated == 794);
}

test "Huge - Bump huge array list" {
    var bump = try Bump.init(1024 * 1024);
    defer bump.deinit();

    var list = std.ArrayList(u32).init(bump.allocator());
    defer list.deinit();

    try list.resize(20);

    try std.testing.expect(bump.allocated == list.capacity * @sizeOf(u32));

    try list.resize(200);

    try std.testing.expect(bump.allocated == list.capacity * @sizeOf(u32));

    list.clearAndFree();

    try std.testing.expect(bump.allocated == 0);
}

test "Huge - Stack ordered frees" {
    var stack = try Stack.init(1024 * 1024);
    var alloc = stack.allocator();
    defer stack.deinit();

    var mem0 = try alloc.alloc(u8, 10);
    var mem1 = try alloc.alloc(u8, 794);

    alloc.free(mem1);
    alloc.free(mem0);

    try std.testing.expect(stack.bump.allocated == 0);
}

test "Huge - Stack unordered frees" {
    var stack = try Stack.init(1024 * 1024);
    var alloc = stack.allocator();
    defer stack.deinit();

    var mem0 = try alloc.alloc(u8, 10);
    var mem1 = try alloc.alloc(u8, 794);

    alloc.free(mem0);
    alloc.free(mem1);

    try std.testing.expect(stack.bump.allocated == mem0.len + 2);
}

test "Huge - Stack reset" {
    var stack = try Stack.init(1024 * 1024);
    var alloc = stack.allocator();
    defer stack.deinit();

    _ = try alloc.alloc(u8, 10);
    _ = try alloc.alloc(u8, 794);

    stack.reset();

    try std.testing.expect(stack.bump.allocated == 0);
}

test "Huge - Stack ordered resize" {
    var stack = try Stack.init(1024 * 1024);
    var alloc = stack.allocator();
    defer stack.deinit();

    var mem0 = try alloc.alloc(u8, 10);
    var mem1 = try alloc.alloc(u8, 794);

    _ = mem0;
    _ = alloc.resize(mem1, mem1.len * 2) orelse unreachable;
}

test "Huge - Freelist ordered frees" {
    var freelist: FreeList = undefined;
    try freelist.init(1024 * 1024);
    var alloc = freelist.allocator();
    defer freelist.deinit();

    var mem0 = try alloc.alloc(u8, 10);
    var mem1 = try alloc.alloc(u8, 794);

    try std.testing.expect(freelist.bump.allocated == 816);

    alloc.free(mem1);
    alloc.free(mem0);

    try std.testing.expect(freelist.bump.allocated == 0);
}
