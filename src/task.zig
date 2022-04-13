const std = @import("std");
const assert = std.debug.assert;

const Semaphore = @import("task/semaphore.zig").Semaphore;
const Queue = @import("task/queue.zig").Queue;
const RwLock = @import("task/rwlock.zig").RwLock;

pub const TaskFn = fn (dipatcher: *Dispatcher, data: *const anyopaque) void;

/// Stores task in a bounded FIFO queue and dispatches them to worker threads for
/// parallel execution. 
pub const Dispatcher = struct {
    /// Provides memory for the queue and workers
    allocator: std.mem.Allocator,
    /// Lock-free FIFO queue where task to be dispatched are stored
    queue: Queue(Task),
    /// Worker threads which process tasks from the queue
    workers: []std.Thread,
    /// Semaphore used to signal work available to the threads, and put them to sleep
    sem: Semaphore,
    /// Number of currently running tasks (up on enqueue, down after completion)
    running: usize,
    /// Flag used to make the worker threads exit
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
        self.sem = .{};
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
