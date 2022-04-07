const std = @import("std");

const buffer = @import("buffer.zig");

//=== Re-exports ===//

/// Coroutine utilities
pub const coro = @import("coro.zig");

/// Huge bump memory allocators
pub const huge = @import("huge.zig");

pub const Buffer = buffer.Buffer;
pub const MemBuffer = buffer.MemBuffer;

pub const task = @import("task.zig");

//=== Common utilities ===//

pub fn FlagsMixin(comptime FlagType: type, comptime IntType: type) type {
    comptime {
        std.debug.assert(@sizeOf(FlagType) == @sizeOf(IntType));
    }

    return struct {
        pub fn toInt(self: FlagType) IntType {
            return @bitCast(IntType, self);
        }

        pub fn fromInt(value: IntType) FlagType {
            return @bitCast(FlagType, value);
        }

        pub fn with(a: FlagType, b: FlagType) FlagType {
            return fromInt(toInt(a) | toInt(b));
        }

        pub fn only(a: FlagType, b: FlagType) FlagType {
            return fromInt(toInt(a) & toInt(b));
        }

        pub fn without(a: FlagType, b: FlagType) FlagType {
            return fromInt(toInt(a) & ~toInt(b));
        }

        pub fn hasAllSet(a: FlagType, b: FlagType) bool {
            return (toInt(a) & toInt(b)) == toInt(b);
        }

        pub fn hasAnySet(a: FlagType, b: FlagType) bool {
            return (toInt(a) & toInt(b)) != 0;
        }

        pub fn isEmpty(a: FlagType) bool {
            return toInt(a) == 0;
        }
    };
}

//=== Testing ===//

test {
    std.testing.refAllDecls(@This());
}
