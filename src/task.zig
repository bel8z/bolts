const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;

const semaphore = @import("task/semaphore.zig");

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

//=== Task dispatch queue ===//

pub const TaskFn = fn (dipatcher: *Dispatcher, data: *const anyopaque) void;

pub const Dispatcher = struct {
    workers: []std.Thread,
    sem: semaphore.Semaphore,
    queue: Queue(Task),
    allocator: std.mem.Allocator,
    running: usize,
    terminate: bool,

    const Self = @This();

    const Task = struct {
        taskfn: TaskFn,
        data: *const anyopaque,
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
        self.sem = semaphore.Semaphore.init();
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

        self.sem.deinit();
    }

    pub fn addTask(self: *Self, taskfn: TaskFn, data: *const anyopaque) bool {
        const task = Task{ .taskfn = taskfn, .data = data };

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

test "Queue" {
    const Context = struct {
        batch_size: usize = 1,
        iter_count: usize = 2000_000,
        start: Atomic(bool) = Atomic(bool).init(false),
        queue: Queue(usize),

        const Self = @This();

        fn threadFn(ctx: *Self) void {
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
    };

    const allocator = std.testing.allocator;

    var ctx = Context{ .queue = try Queue(usize).alloc(allocator, 1024) };
    defer ctx.queue.free(allocator);

    var iter: usize = 0;
    while (iter < 100) : (iter += 1) {
        try std.testing.expect(ctx.queue.dequeue() == null);
    }

    var threads: [4]std.Thread = undefined;
    for (threads) |*thread| {
        thread.* = try std.Thread.spawn(.{}, Context.threadFn, .{&ctx});
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

fn testTaskFn(_: *Dispatcher, data: *const anyopaque) void {
    var str = @ptrCast([*:0]const u8, data);
    std.debug.print("{s}\n", .{str});
}

test "Task" {
    const allocator = std.testing.allocator;
    var d: Dispatcher = undefined;
    try d.init(allocator, 0, 0);
    defer d.deinit();

    std.debug.print("\n", .{});

    try std.testing.expect(d.addTask(testTaskFn, "String A0"));
    try std.testing.expect(d.addTask(testTaskFn, "String A1"));
    try std.testing.expect(d.addTask(testTaskFn, "String A2"));
    try std.testing.expect(d.addTask(testTaskFn, "String A3"));
    try std.testing.expect(d.addTask(testTaskFn, "String A4"));
    try std.testing.expect(d.addTask(testTaskFn, "String A5"));
    try std.testing.expect(d.addTask(testTaskFn, "String A6"));
    try std.testing.expect(d.addTask(testTaskFn, "String A7"));
    try std.testing.expect(d.addTask(testTaskFn, "String A8"));
    try std.testing.expect(d.addTask(testTaskFn, "String A9"));

    try std.testing.expect(d.addTask(testTaskFn, "String B0"));
    try std.testing.expect(d.addTask(testTaskFn, "String B1"));
    try std.testing.expect(d.addTask(testTaskFn, "String B2"));
    try std.testing.expect(d.addTask(testTaskFn, "String B3"));
    try std.testing.expect(d.addTask(testTaskFn, "String B4"));
    try std.testing.expect(d.addTask(testTaskFn, "String B5"));
    try std.testing.expect(d.addTask(testTaskFn, "String B6"));
    try std.testing.expect(d.addTask(testTaskFn, "String B7"));
    try std.testing.expect(d.addTask(testTaskFn, "String B8"));
    try std.testing.expect(d.addTask(testTaskFn, "String B9"));

    d.waitCompletion();
}

test {
    std.testing.refAllDecls(@This());
}
