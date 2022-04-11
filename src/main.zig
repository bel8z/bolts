const std = @import("std");
const bolts = @import("bolts");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var allocator = gpa.allocator();
    defer _ = gpa.deinit();

    var d: bolts.task.Dispatcher = undefined;
    try d.init(allocator, 1, 256);
    defer d.deinit();
}
