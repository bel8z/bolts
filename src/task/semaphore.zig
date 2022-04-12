const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;

const win32 = std.os.windows;

const task = @import("../task.zig");

pub const Semaphore = LightSemaphore;

/// Lightweight semaphore that uses a spin wait to reduce calls into the underlying
/// OS object.
/// Based on https://preshing.com/20150316/semaphores-are-surprisingly-versatile
const LightSemaphore = struct {
    impl: Impl,
    count: i32,

    const Self = @This();
    const Impl = if (builtin.os.tag == .windows) Win32Semaphore else StdSemaphore;

    pub fn init() Self {
        return Self{
            .impl = Impl.init(),
            .count = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        self.impl.deinit();
    }

    pub fn wait(self: *Self) void {
        var prev_count = @atomicLoad(i32, &self.count, .Monotonic);

        var spin: usize = 0;
        while (spin < 10_000) : (spin += 1) {
            if (prev_count > 0) {
                if (@cmpxchgStrong(
                    i32,
                    &self.count,
                    prev_count,
                    prev_count - 1,
                    .Acquire,
                    .Acquire,
                )) |updated| {
                    prev_count = updated;
                } else {
                    return;
                }
                // Prevent the compiler from collapsing the loop.
                std.atomic.compilerFence(.Acquire);
            }
        }

        prev_count = @atomicRmw(i32, &self.count, .Sub, 1, .Acquire);
        if (prev_count <= 0) self.impl.wait();
    }

    pub fn post(self: *Self) void {
        const prev_count = @atomicRmw(i32, &self.count, .Add, 1, .Release);
        const release = if (-prev_count < 1) -prev_count else 1;
        if (release > 0) {
            self.impl.post();
        }
    }
};

const StdSemaphore = struct {
    sem: std.Thread.Semaphore = .{},

    const Self = @This();

    pub fn init() Self {
        return Self{};
    }

    pub fn deinit(self: *Self) void {
        _ = self;
    }

    pub fn wait(self: *Self) void {
        self.sem.wait();
    }

    pub fn post(self: *Self) void {
        self.sem.post();
    }
};

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

fn Benaphore(comptime Sem: type) type {
    return struct {
        contention_count: i32,
        sem: Sem,

        const Self = @This();

        pub fn init() Self {
            return Self{
                .sem = Sem.init(),
                .contention_count = 0,
            };
        }

        pub fn lock(self: *Self) void {
            if (@atomicRmw(i32, &self.contention_count, .Add, 1, .Acquire) > 0) {
                self.sem.wait();
            }
        }

        pub fn tryLock(self: *Self) bool {
            if (@atomicLoad(i32, &self.contention_count, .Monotonic) != 0) {
                return false;
            }

            return (@cmpxchgStrong(
                i32,
                &self.contention_count,
                0,
                1,
                .Acquire,
                .Acquire,
            ) == null);
        }

        pub fn unlock(self: *Self) void {
            const prev_count = @atomicRmw(i32, &self.contention_count, .Sub, 1, .Release);
            std.debug.assert(prev_count > 0);
            if (prev_count > 1) self.sem.post();
        }
    };
}

test "Benaphore" {
    const thread_count = 4;
    const iter_count = 400_000;

    const Mutex = Benaphore(StdSemaphore);

    const Context = struct {
        value: i32,
        mutex: Mutex,

        fn threadFn(self: *@This(), id: usize) void {
            _ = id;

            var iter: u32 = 0;
            while (iter < iter_count) : (iter += 1) {
                self.mutex.lock();
                defer self.mutex.unlock();

                self.value += 1;
            }
        }
    };

    var context = Context{
        .value = 0,
        .mutex = Mutex.init(),
    };

    var threads: [thread_count]std.Thread = undefined;

    for (threads) |*thread, i| {
        thread.* = try std.Thread.spawn(.{}, Context.threadFn, .{ &context, i });
    }

    for (threads) |thread| {
        thread.join();
    }

    try std.testing.expect(context.value == iter_count * thread_count);
}
