const std = @import("std");

pub const Channel = @import("channel.zig").Channel;

test {
    std.testing.refAllDecls(@This());

    _ = @import("channel.zig");
}
