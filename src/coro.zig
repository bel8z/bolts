const std = @import("std");
const testing = std.testing;
const assert = std.debug.assert;

/// Implements a communication channel between coroutines, in order to allow 
/// them to share data and synchronize via message passing.
/// It can be used as a building block for generators.
pub fn Channel(comptime T: type) type {
    return struct {
        const Self = @This();
        value: ?T = null,
        waiting_recv: ?anyframe = null,
        waiting_send: ?anyframe = null,

        /// Sends a value to the channel.
        /// Suspends until the value is fetched by a call to recv.
        pub fn send(self: *Self, t: T) void {
            assert(self.value == null);
            self.value = t;
            if (self.waiting_recv) |recv_op| {
                resume recv_op;
            } else {
                suspend {
                    self.waiting_send = @frame();
                }
                self.waiting_send = null;
            }
        }

        /// Returns the value in the channel, or if there is no value present,
        /// suspends until a value is sent into the channel.
        pub fn recv(self: *Self) T {
            if (self.value) |val| {
                self.value = null;
                assert(self.waiting_send != null);
                resume self.waiting_send.?;
                return val;
            } else {
                suspend {
                    self.waiting_recv = @frame();
                }
                self.waiting_recv = null;

                assert(self.waiting_send == null);
                assert(self.value != null);
                const val = self.value.?;
                self.value = null;
                return val;
            }
        }
    };
}

fn fibonacciGenerator(channel: *Channel(?i64), max: i64) void {
    var a: i64 = 0;
    var b: i64 = 1;
    while (b <= max) {
        channel.send(b);
        const next = a + b;
        a = b;
        b = next;
    }
    channel.send(null);
}

fn testFibonacciGenerator(finished_test: *bool) !void {
    var channel = Channel(?i64){};
    var frame = async fibonacciGenerator(&channel, 13);
    try testing.expect(channel.recv().? == @intCast(i64, 1));
    try testing.expect(channel.recv().? == @intCast(i64, 1));
    try testing.expect(channel.recv().? == @intCast(i64, 2));
    try testing.expect(channel.recv().? == @intCast(i64, 3));
    try testing.expect(channel.recv().? == @intCast(i64, 5));
    try testing.expect(channel.recv().? == @intCast(i64, 8));
    try testing.expect(channel.recv().? == @intCast(i64, 13));
    try testing.expect(channel.recv() == null);
    await frame;
    finished_test.* = true;
}

test "Channel - Fibonacci generator" {
    // NOTE This pattern is required for testing coroutines
    var finished_test = false;
    _ = async testFibonacciGenerator(&finished_test);
    try testing.expect(finished_test);
}
