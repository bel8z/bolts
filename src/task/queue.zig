const std = @import("std");

const Atomic = std.atomic.Atomic;

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
        // TODO (Matteo):
        // * Use a cache size based on the actual target platform
        // * The layout can be improved to reduce size a bit?
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
