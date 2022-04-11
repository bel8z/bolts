const std = @import("std");
const builtin = @import("builtin");
const win32 = std.os.windows;
const assert = std.debug.assert;

const Atomic = std.atomic.Atomic;

//=== Building blocks ===//

/// Multi producer, multi consumer, bounded wait-free queue
/// Fails on overflow, does not overwrite old data.
/// Based on https://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue
pub fn Queue(comptime T: type) type {
    return extern struct {
        _pad0: CachelinePadding = undefined,
        enqueue_pos: Atomic(usize),
        _pad1: CachelinePadding = undefined,
        dequeue_pos: Atomic(usize),
        _pad2: CachelinePadding = undefined,
        buffer: [*]BufferSlot,
        buffer_mask: usize,

        const Self = @This();

        // Padding is used between fields in order to place them in different
        // cache lines and avoid false sharing
        const CachelinePadding = [64]u8;

        /// Couples data and a sequence counter, used to detect overflow and order enqueues
        const BufferSlot = struct {
            sequence: Atomic(usize),
            data: T,
        };

        pub fn alloc(allocator: std.mem.Allocator, size: usize) !Self {
            if (!std.math.isPowerOfTwo(size)) return error.SizeNotPowerOfTwo;

            const buffer = try allocator.alloc(BufferSlot, size);

            for (buffer) |*cell, i| {
                cell.sequence = Atomic(usize).init(i);
            }

            return Self{
                .buffer = buffer.ptr,
                .buffer_mask = buffer.len - 1,
                .enqueue_pos = Atomic(usize).init(0),
                .dequeue_pos = Atomic(usize).init(0),
            };
        }

        pub fn free(self: *Self, allocator: std.mem.Allocator) void {
            const buffer = self.buffer[0 .. self.buffer_mask + 1];
            allocator.free(buffer);
        }

        pub fn enqueue(self: *Self, data: T) bool {
            var curr_pos = self.enqueue_pos.load(.Monotonic);

            while (true) {
                var slot = &self.buffer[curr_pos & self.buffer_mask];

                const next_pos = curr_pos +% 1;
                const sequence = slot.sequence.load(.Acquire);
                const diff = std.math.sub(usize, sequence, curr_pos) catch return false;

                if (diff > 0) {
                    curr_pos = self.enqueue_pos.load(.Monotonic);
                } else if (self.enqueue_pos.tryCompareAndSwap(curr_pos, next_pos, .Monotonic, .Monotonic)) |changed_pos| {
                    curr_pos = changed_pos;
                } else {
                    slot.data = data;
                    slot.sequence.store(next_pos, .Release);
                    return true;
                }
            }
        }

        pub fn dequeue(self: *Self) ?T {
            var curr_pos = self.dequeue_pos.load(.Monotonic);

            while (true) {
                var slot = &self.buffer[curr_pos & self.buffer_mask];

                const next_pos = curr_pos +% 1;
                const sequence = slot.sequence.load(.Acquire);
                const diff = std.math.sub(usize, sequence, next_pos) catch return null;

                if (diff > 0) {
                    curr_pos = self.dequeue_pos.load(.Monotonic);
                } else if (self.dequeue_pos.tryCompareAndSwap(curr_pos, next_pos, .Monotonic, .Monotonic)) |changed_pos| {
                    curr_pos = changed_pos;
                } else {
                    slot.sequence.store(next_pos +% self.buffer_mask, .Release);
                    return slot.data;
                }
            }
        }
    };
}

const Win32Semaphore = struct {
    handle: win32.HANDLE,

    const Self = @This();

    pub fn init() Self {
        return Self{
            .handle = CreateSemaphoreA(
                null,
                @intCast(i32, 0),
                std.math.maxInt(win32.LONG),
                null,
            ),
        };
    }

    pub fn deinit(self: *Self) void {
        win32.CloseHandle(self.handle);
    }

    pub fn wait(self: *Self) void {
        win32.WaitForSingleObject(self.handle, win32.INFINITE) catch unreachable;
    }

    pub fn post(self: *Self) void {
        var prev_count: i32 = 0;
        const done = ReleaseSemaphore(self.handle, 1, &prev_count);
        assert(done);
    }

    extern "kernel32" fn CreateSemaphoreA(
        attributes: ?*anyopaque,
        initial_count: win32.LONG,
        maximum_count: win32.LONG,
        name: ?[*:0]const u8,
    ) callconv(win32.WINAPI) win32.HANDLE;

    extern "kernel32" fn ReleaseSemaphore(
        hSemaphore: win32.HANDLE,
        lReleaseCount: win32.LONG,
        lpPreviousCount: *win32.LONG,
    ) callconv(win32.WINAPI) bool;
};

//=== Task dispatch queue ===//

pub const TaskFn = fn (dipatcher: *Dispatcher, data: *anyopaque) void;

pub const Dispatcher = struct {
    workers: []std.Thread,
    sem: Semaphore,
    queue: Queue(Task),
    allocator: std.mem.Allocator,
    running: usize,
    terminate: bool,

    const Self = @This();

    const Semaphore = if (builtin.os.tag == .windows) Win32Semaphore else std.Thread.Semaphore;

    const Task = struct {
        taskfn: TaskFn,
        data: *anyopaque,
    };

    pub fn init(
        self: *Self,
        allocator: std.mem.Allocator,
        worker_count: usize,
        buffer_size: usize,
    ) !void {
        const w = if (worker_count > 0) worker_count else (try std.Thread.getCpuCount()) - 1;
        const q = if (buffer_size > 0) buffer_size else 256;

        self.allocator = allocator;
        self.running = 0;
        self.terminate = false;
        self.sem = if (Semaphore == Win32Semaphore) Semaphore.init() else .{};
        self.queue = try Queue(Task).alloc(allocator, q);
        self.workers = try allocator.alloc(std.Thread, w);

        for (self.workers) |*worker, i| {
            worker.* = try std.Thread.spawn(.{}, threadFn, .{ self, i });
        }
    }

    pub fn deinit(self: *Self) void {
        self.waitCompletion();

        @atomicStore(bool, &self.terminate, true, .SeqCst);

        for (self.workers) |_| {
            self.sem.post();
        }

        for (self.workers) |worker| {
            worker.join();
        }

        self.allocator.free(self.workers);
        self.queue.free(self.allocator);

        if (Semaphore == Win32Semaphore) {
            self.sem.deinit();
        }
    }

    pub fn addTask(self: *Self, taskfn: TaskFn, data: anytype) bool {
        const task = Task{ .taskfn = taskfn, .data = @ptrCast(*anyopaque, data) };

        if (self.queue.enqueue(task)) {
            _ = @atomicRmw(usize, &self.running, .Add, 1, .Release);
            self.sem.post();
            return true;
        }
        return false;
    }

    pub fn waitCompletion(self: *Self) void {
        while (@atomicLoad(usize, &self.running, .Acquire) > 0) {
            _ = self.performTask();
        }
    }

    fn performTask(self: *Self) bool {
        if (self.queue.dequeue()) |task| {
            task.taskfn(self, task.data);

            const prev_running = @atomicRmw(usize, &self.running, .Sub, 1, .Release);
            assert(prev_running > 0);

            return true;
        }

        return false;
    }

    fn threadFn(self: *Self, thread_index: usize) void {
        _ = thread_index;

        while (!@atomicLoad(bool, &self.terminate, .SeqCst)) {
            if (!self.performTask()) self.sem.wait();
        }
    }
};

//=== Testing ===//

const TestQueueContext = struct {
    batch_size: usize = 1,
    iter_count: usize = 2000_000,
    start: Atomic(bool) = Atomic(bool).init(false),
    queue: Queue(usize),
};

fn testQueueFn(ctx: *TestQueueContext) void {
    const yield = std.os.windows.kernel32.SwitchToThread;

    const id = std.Thread.getCurrentId();

    const seed = @intCast(u64, std.time.milliTimestamp()) + id;
    var prng = std.rand.DefaultPrng.init(seed);
    var pause = 1 + prng.random().int(usize) % 1000;

    while (!ctx.start.load(.Acquire)) {
        _ = yield();
    }

    while (pause > 1) : (pause -= 1) {
        std.atomic.spinLoopHint();
    }

    var iter: usize = 0;
    while (iter < ctx.iter_count) : (iter += 1) {
        var batch: usize = undefined;

        batch = 0;
        while (batch < ctx.batch_size) : (batch += 1) {
            while (!ctx.queue.enqueue(batch)) _ = yield();
        }

        batch = 0;
        while (batch < ctx.batch_size) : (batch += 1) {
            while (true) {
                if (ctx.queue.dequeue()) |_| break;
                _ = yield();
            }
        }
    }
}

test "Queue" {
    const allocator = std.testing.allocator;

    var ctx = TestQueueContext{ .queue = try Queue(usize).alloc(allocator, 1024) };
    defer ctx.queue.free(allocator);

    var iter: usize = 0;
    while (iter < 100) : (iter += 1) {
        try std.testing.expect(ctx.queue.dequeue() == null);
    }

    var threads: [4]std.Thread = undefined;
    for (threads) |*thread| {
        thread.* = try std.Thread.spawn(.{}, testQueueFn, .{&ctx});
    }

    std.time.sleep(1000_1000);

    const timer = try std.time.Timer.start();

    ctx.start.store(true, .Release);

    for (threads) |thread| {
        thread.join();
    }

    const time = timer.read();
    const ops = ctx.batch_size * ctx.iter_count * 2 * threads.len;
    const freq = @intToFloat(f64, time) / @intToFloat(f64, ops);

    std.debug.print("{d} nanosecond per operation\n", .{freq});
}

const TestTaskContext = struct {
    str: []const u8,
};

fn testTaskFn(_: *Dispatcher, data: *anyopaque) void {
    var ctx = @ptrCast(*TestTaskContext, @alignCast(@alignOf(TestTaskContext), data));
    std.debug.print("\n{s}\n", .{ctx.str});
}

test "Task" {
    const allocator = std.testing.allocator;
    var d: Dispatcher = undefined;
    try d.init(allocator, 0, 0);
    defer d.deinit();

    var ctx = TestTaskContext{ .str = "YEAH" };

    if (d.addTask(testTaskFn, &ctx)) {
        d.waitCompletion();
    }
}
