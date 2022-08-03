const std = @import("std");
const math = std.math;
const assert = std.debug.assert;

pub usingnamespace math;

// Utilities for float comparisons
const eps = 1e-5;
const epsSq = 1e-10;

// TODO (Matteo):
// * Equality (also approximate with epsilon)
// * Shear transform
// * Review camera (view) matrices (LookAt, LookTo)

/// 2D Vector of 32 bit integers
pub const Vec2I = Vec2(i32, DefaultTag);

/// 2D Vector of single-precision (32 bit) float values
pub const Vec2S = Vec2(f32, DefaultTag);

/// 2D Vector of double -precision (64 bit) float values
pub const Vec2D = Vec2(f64, DefaultTag);

/// 3D Vector of 33-bit integers
pub const Vec3I = Vec3(i32, DefaultTag);

/// 3D Vector of single-precision (32 bit) float values
pub const Vec3S = Vec3(f32, DefaultTag);

/// 3D Vector of double -precision (64 bit) float values
pub const Vec3D = Vec3(f64, DefaultTag);

/// 4D Vector of 34-bit integers
pub const Vec4I = Vec4(i32, DefaultTag);

/// 4D Vector of single-precision (32 bit) float values
pub const Vec4S = Vec4(f32, DefaultTag);

/// 4D Vector of double -precision (64 bit) float values
pub const Vec4D = Vec4(f64, DefaultTag);

/// 4x4 matrix of single-precision (32 bit) float values
pub const Mat4S = Mat4(f32, DefaultTag);

/// 4x4 matrix of double -precision (64 bit) float values
pub const Mat4D = Mat4(f64, DefaultTag);

/// Default tag struct for providing common vector types
const DefaultTag = struct {};

/// 2D vector of generic scalar values
pub fn Vec2(comptime Scalar: type, comptime Tag: type) type {
    return extern struct {
        x: Scalar,
        y: Scalar,

        const Self = @This();

        comptime {
            std.debug.assert(@sizeOf(Tag) == 0);
            std.debug.assert(@sizeOf(Self) == 2 * @sizeOf(Scalar));
        }

        pub usingnamespace VecMixin(Scalar, Self);

        pub const zero = init(0, 0);
        pub const x_axis = init(1, 0);
        pub const y_axis = init(0, 1);

        pub inline fn init(x: Scalar, y: Scalar) Self {
            return .{ .x = x, .y = y };
        }

        pub fn negate(v: Self) Self {
            return .{
                .x = -v.x,
                .y = -v.y,
            };
        }

        pub fn dot(lhs: Self, rhs: Self) Scalar {
            return lhs.x * rhs.x + lhs.y * rhs.y;
        }

        pub fn perpDot(lhs: Self, rhs: Self) Scalar {
            return lhs.x * rhs.y - lhs.y * rhs.x;
        }

        pub fn add(lhs: Self, rhs: Self) Self {
            return .{
                .x = lhs.x + rhs.x,
                .y = lhs.y + rhs.y,
            };
        }

        pub fn sub(lhs: Self, rhs: Self) Self {
            return .{
                .x = lhs.x - rhs.x,
                .y = lhs.y - rhs.y,
            };
        }

        pub fn mul(lhs: Self, rhs: Scalar) Self {
            return .{
                .x = lhs.x * rhs,
                .y = lhs.y * rhs,
            };
        }

        pub fn div(lhs: Self, rhs: Scalar) Self {
            return .{
                .x = lhs.x / rhs,
                .y = lhs.y / rhs,
            };
        }
    };
}

// 3D vector of generic scalar values
pub fn Vec3(comptime Scalar: type, comptime Tag: type) type {
    return extern struct {
        x: Scalar,
        y: Scalar,
        z: Scalar,

        const Self = @This();

        comptime {
            std.debug.assert(@sizeOf(Tag) == 0);
            std.debug.assert(@sizeOf(Self) == 3 * @sizeOf(Scalar));
        }

        pub usingnamespace VecMixin(Scalar, Self);

        pub const zero = init(0, 0, 0);
        pub const x_axis = init(1, 0, 0);
        pub const y_axis = init(0, 1, 0);
        pub const z_axis = init(0, 0, 1);

        pub inline fn init(x: Scalar, y: Scalar, z: Scalar) Self {
            return .{ .x = x, .y = y, .z = z };
        }

        pub fn negate(v: Self) Self {
            return .{
                .x = -v.x,
                .y = -v.y,
                .z = -v.z,
            };
        }

        pub fn dot(lhs: Self, rhs: Self) Scalar {
            return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
        }

        pub fn add(lhs: Self, rhs: Self) Self {
            return .{
                .x = lhs.x + rhs.x,
                .y = lhs.y + rhs.y,
                .z = lhs.z + rhs.z,
            };
        }

        pub fn sub(lhs: Self, rhs: Self) Self {
            return .{
                .x = lhs.x - rhs.x,
                .y = lhs.y - rhs.y,
                .z = lhs.z - rhs.z,
            };
        }

        pub fn mul(lhs: Self, rhs: Scalar) Self {
            return .{
                .x = lhs.x * rhs,
                .y = lhs.y * rhs,
                .z = lhs.z * rhs,
            };
        }

        pub fn div(lhs: Self, rhs: Scalar) Self {
            return .{
                .x = lhs.x / rhs,
                .y = lhs.y / rhs,
                .z = lhs.z / rhs,
            };
        }

        pub fn cross(a: Self, b: Self) Self {
            return .{
                .x = a.y * b.z - a.z * b.y,
                .y = a.z * b.x - a.x * b.z,
                .z = a.x * b.y - a.y * b.x,
            };
        }
    };
}

// 4D vector of generic scalar values
pub fn Vec4(comptime Scalar: type, comptime Tag: type) type {
    return extern struct {
        x: Scalar,
        y: Scalar,
        z: Scalar,
        w: Scalar,

        const Self = @This();

        comptime {
            std.debug.assert(@sizeOf(Tag) == 0);
            std.debug.assert(@sizeOf(Self) == 4 * @sizeOf(Scalar));
        }

        pub usingnamespace VecMixin(Scalar, Self);

        pub const zero = init(0, 0, 0, 0);

        pub inline fn init(x: Scalar, y: Scalar, z: Scalar, w: Scalar) Self {
            return .{ .x = x, .y = y, .z = z, .w = w };
        }

        pub fn negate(v: Self) Self {
            return .{
                .x = -v.x,
                .y = -v.y,
                .z = -v.z,
                .w = -v.w,
            };
        }

        pub fn dot(lhs: Self, rhs: Self) Scalar {
            return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z + lhs.w * rhs.w;
        }

        pub fn add(lhs: Self, rhs: Self) Self {
            return .{
                .x = lhs.x + rhs.x,
                .y = lhs.y + rhs.y,
                .z = lhs.z + rhs.z,
                .w = lhs.w + rhs.w,
            };
        }

        pub fn sub(lhs: Self, rhs: Self) Self {
            return .{
                .x = lhs.x - rhs.x,
                .y = lhs.y - rhs.y,
                .z = lhs.z - rhs.z,
                .w = lhs.w - rhs.w,
            };
        }

        pub fn mul(lhs: Self, rhs: Scalar) Self {
            return .{
                .x = lhs.x * rhs,
                .y = lhs.y * rhs,
                .z = lhs.z * rhs,
                .w = lhs.w * rhs,
            };
        }

        pub fn div(lhs: Self, rhs: Scalar) Self {
            return .{
                .x = lhs.x / rhs,
                .y = lhs.y / rhs,
                .z = lhs.z / rhs,
                .w = lhs.w / rhs,
            };
        }
    };
}

/// Mixin that provides vector operations common across multiple dimensions
/// based on specific ones (e.g. 'norm' based on 'dot') 
fn VecMixin(comptime ScalarT: type, comptime Self: type) type {
    return struct {

        /// Re-exported scalar type
        pub const Scalar = ScalarT;

        /// Compute the squared norm of the vector
        pub inline fn normSq(v: Self) Scalar {
            return v.dot(v);
        }

        /// Compute the norm (length) of the vector
        pub inline fn norm(v: Self) Scalar {
            return math.sqrt(v.normSq());
        }

        /// Compute the normalized form of the vector (scale to unit length)
        pub inline fn normalized(v: Self) Self {
            const norm2 = v.normSq();

            // NOTE (Matteo): Save some operations for the unit and zero vectors
            if (math.approxEqAbs(Scalar, norm2, 1, epsSq) or
                math.approxEqAbs(Scalar, norm2, 0, epsSq))
            {
                return v;
            }

            return v.mul(1 / math.sqrt(norm2));
        }
    };
}

/// Target clip space for projection matrices.
/// All coordinates up to view space are considered right-handed, but clip space
/// is generally left-handed.
/// The range of the Z axis also varies between graphics APIs, so we specify the 
/// clip space as a struct instead of having different functions for different spaces.
pub const ClipSpace = packed struct {
    y_dir: i3,
    z_near: i3,
    z_far: i3,

    pub fn init(y_dir: i3, z_near: i3, z_far: i3) ClipSpace {
        assert(y_dir != 0);
        assert(z_near != 0 or z_far != 0);
        assert(z_near != z_far);

        return ClipSpace{ .y_dir = y_dir, .z_near = z_near, .z_far = z_far };
    }

    // TODO (Matteo): Experiment more with reverse depth
    // Note that in OpenGL the following is required in order for the depth buffer
    // to work correctly:
    // - glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE)
    // - glDepthFunc(GL_GEQUAL) since now the closest values to the camera have an increasing Z value
    // - clear the depth buffer to 0 instead of the default 1, because of the inverted range

    // NOTE (Matteo): OpenGL uses by (legacy) convention a left-handed clip space, with all coordinates
    // normalized in the [-1, 1] interval, Z-included.
    pub const opengl = init(1, -1, 1);

    // NOTE (Matteo): Direct3D uses by convention a left-handed clip space, with the Z coordinate
    // normalized in the [0, 1] interval
    pub const d3d = init(1, 0, 1);

    pub fn vulkan(reverse_depth: bool) ClipSpace {
        // NOTE (Matteo): Vulkan uses a left-handed clip space, with the origin in the top left corner,
        // Y axis positive downwards and Z coordinate normalized by default in the [0, 1] interval.
        // According to https://vincent-p.github.io/posts/vulkan_perspective_matrix, using
        // "reverse depth" gives better numerical distribution so we accept it as a parameter.
        var z_near: i3 = 0;
        var z_far: i3 = 0;

        if (reverse_depth) {
            z_far = 1;
        } else {
            z_near = 1;
        }

        return init(-1, z_near, z_far);
    }
};

/// 4x4 matrix of generic scalar values
/// The chosen representation is column-major for interoperability with graphics 
/// APIs and shader code.
pub fn Mat4(comptime Scalar: type, comptime Tag: type) type {
    return extern struct {
        elem: [4][4]Scalar,

        const Self = @This();
        const V = std.meta.Vector(4, Scalar);

        comptime {
            std.debug.assert(@sizeOf(Tag) == 0);
            std.debug.assert(@sizeOf(Self) == @sizeOf([4][4]Scalar));
        }

        //=== Basic construction ===//

        /// Represents the identity matrix
        pub const identity: Self = diag(1);

        /// Represents a matrix of all 0 values
        pub const zero: Self = diag(0);

        /// Construct a diagonal matrix
        pub inline fn diag(value: Scalar) Self {
            return Self{
                .elem = .{
                    .{ value, 0, 0, 0 },
                    .{ 0, value, 0, 0 },
                    .{ 0, 0, value, 0 },
                    .{ 0, 0, 0, value },
                },
            };
        }

        /// Construct the matrix given its 4 columns
        pub fn fromColumns(a: anytype, b: anytype, c: anytype, d: anytype) Self {
            const T = @TypeOf(a, b, c, d);

            switch (T) {
                [4]Scalar => {
                    return Self{ .elem = .{ a, b, c, d } };
                },
                Vec4(Scalar, Tag) => {
                    return Self{ .elem = .{
                        .{ a.x, a.y, a.z, a.w },
                        .{ b.x, b.y, b.z, b.w },
                        .{ c.x, c.y, c.z, c.w },
                        .{ d.x, d.y, d.z, d.w },
                    } };
                },
                else => {
                    @compileError("Matrix initialization not implemented for " ++ @typeName(T));
                },
            }
        }

        /// Construct the matrix given its 4 rows
        pub fn fromRows(a: anytype, b: anytype, c: anytype, d: anytype) Self {
            const T = @TypeOf(a, b, c, d);

            switch (T) {
                [4]Scalar => {
                    return Self{ .elem = .{
                        .{ a[0], b[0], c[0], d[0] },
                        .{ a[1], b[1], c[1], d[1] },
                        .{ a[2], b[2], c[2], d[2] },
                        .{ a[3], b[3], c[3], d[3] },
                    } };
                },
                Vec4(Scalar, Tag) => {
                    return Self{ .elem = .{
                        .{ a.x, b.x, c.x, d.x },
                        .{ a.y, b.y, c.y, d.y },
                        .{ a.z, b.z, c.z, d.z },
                        .{ a.w, b.w, c.w, d.w },
                    } };
                },
                else => {
                    @compileError("Matrix initialization not implemented for " ++ @typeName(T));
                },
            }
        }

        //=== Common transforms ===//

        pub fn scale(x: Scalar, y: Scalar, z: Scalar) Self {
            return Self{ .elem = .{
                .{ x, 0, 0, 0 },
                .{ 0, y, 0, 0 },
                .{ 0, 0, z, 0 },
                .{ 0, 0, 0, 1 },
            } };
        }

        pub inline fn scaleVec(v: anytype) Self {
            const T = @TypeOf(v);

            switch (T) {
                Vec2(Scalar, Tag) => return scale(v.x, v.y, 0),
                Vec3(Scalar, Tag) => return scale(v.x, v.y, v.z),
                Vec4(Scalar, Tag) => return scale(v.x, v.y, v.z, v.w),
                else => @compileError("Type not supported " ++ @typeName(T)),
            }
        }

        pub fn translation(x: Scalar, y: Scalar, z: Scalar) Self {
            return Self{ .elem = .{
                .{ 1, 0, 0, 0 },
                .{ 0, 1, 0, 0 },
                .{ 0, 0, 1, 0 },
                .{ x, y, z, 1 },
            } };
        }

        pub inline fn translationVec(v: anytype) Self {
            const T = @TypeOf(v);

            switch (T) {
                Vec2(Scalar, Tag) => return translation(v.x, v.y, 0),
                Vec3(Scalar, Tag) => return translation(v.x, v.y, v.z),
                Vec4(Scalar, Tag) => return translation(v.x, v.y, v.z),
                else => @compileError("Type not supported " ++ @typeName(T)),
            }
        }

        pub fn rotation(axis: Vec3(Scalar, Tag), radians: Scalar) Self {
            const cost = math.cos(radians);
            const sint = math.sin(radians);
            const h = (1 - cost);

            const axis_norm = axis.normalized();
            const rx: Scalar = axis_norm.x;
            const ry: Scalar = axis_norm.y;
            const rz: Scalar = axis_norm.z;

            var m = identity;

            m.elem[0][0] = cost + rx * rx * h;
            m.elem[1][0] = ry * rx * h - rz * sint;
            m.elem[2][0] = rz * rx * h + ry * sint;

            m.elem[0][1] = rx * ry * h + rz * sint;
            m.elem[1][1] = cost + ry * ry * h;
            m.elem[2][1] = rz * ry * h - rx * sint;

            m.elem[0][2] = rx * rz * h - ry * sint;
            m.elem[1][2] = ry * rz * h + rx * sint;
            m.elem[2][2] = cost + rz * rz * h;

            return m;
        }

        pub fn rotationX(radians: Scalar) Self {
            const cost = math.cos(radians);
            const sint = math.sin(radians);

            return Self{ .elem = .{
                .{ 1, 0, 0, 0 },
                .{ 0, cost, sint, 0 },
                .{ 0, -sint, cost, 0 },
                .{ 0, 0, 0, 1 },
            } };
        }

        pub fn rotationY(radians: Scalar) Self {
            const cost = math.cos(radians);
            const sint = math.sin(radians);

            return Self{ .elem = .{
                .{ cost, 0, -sint, 0 },
                .{ 0, 1, 0, 0 },
                .{ sint, 0, cost, 0 },
                .{ 0, 0, 0, 1 },
            } };
        }

        pub fn rotationZ(radians: Scalar) Self {
            const cost = math.cos(radians);
            const sint = math.sin(radians);

            return Self{ .elem = .{
                .{ cost, sint, 0, 0 },
                .{ -sint, cost, 0, 0 },
                .{ 0, 0, 1, 0 },
                .{ 0, 0, 0, 1 },
            } };
        }

        //=== Views ===//

        /// Look from 'position' towards 'target', with the given 'up' direction
        pub fn lookAt(
            position: Vec3(Scalar, Tag),
            target: Vec3(Scalar, Tag),
            up: Vec3(Scalar, Tag),
        ) Self {
            return look(position, position.sub(target), up);
        }

        /// Same as 'lookAt' but uses the Y axis as the 'up' direction
        pub fn lookAtYup(
            position: Vec3(Scalar, Tag),
            target: Vec3(Scalar, Tag),
        ) Self {
            return look(position, position.sub(target), Vec3(Scalar, Tag).y_axis);
        }

        /// Look from 'position' along the given 'direction', with the given 'up' direction
        /// Beware that 'direction' is positive towards 'position', not the target 
        pub fn look(
            position: Vec3(Scalar, Tag),
            direction: Vec3(Scalar, Tag),
            up: Vec3(Scalar, Tag),
        ) Self {
            const z = direction.normalized();
            const x = up.cross(z).normalized();
            const y = z.cross(x);

            var m = identity;
            // First row - camera X axis
            m.elem[0][0] = x.x;
            m.elem[1][0] = x.y;
            m.elem[2][0] = x.z;
            // Second row - camera Y axis
            m.elem[0][1] = y.x;
            m.elem[1][1] = y.y;
            m.elem[2][1] = y.z;
            // Third row - camera Z axis
            m.elem[0][2] = z.x;
            m.elem[1][2] = z.y;
            m.elem[2][2] = z.z;
            // Fourth column - camera translation component obtained by multiplying
            // a matrix defined as above with the matrix that represents translation 't'
            const t = position.negate();
            m.elem[3][0] = x.dot(t);
            m.elem[3][1] = y.dot(t);
            m.elem[3][2] = z.dot(t);

            return m;
        }

        //=== Projections ===//

        // NOTE (Matteo): Currently projections transform from a RH space
        // to a LH clip space (namely OpenGL NDC)
        // TODO (Matteo): Handle different coordinate systems (i.e. clip spaces)

        /// Build a symmetrical perspective projection matrix given the field of 
        /// view and aspect ratio
        pub fn perspectiveFov(
            fovy: Scalar, // field of view along the y axis, in radians
            aspect: Scalar, // aspect ratio (width / height)
            near: Scalar, // distance of the near plane from the camera (along positive Z)
            far: Scalar, // distance of the far plane from the camera (along positive Z)
            clip: ClipSpace,
        ) Self {
            assert(!math.approxEqAbs(Scalar, aspect, 0, eps));

            // Compute the focal  length from the given field of view
            const tanf = math.tan(0.5 * fovy);
            assert(!math.approxEqAbs(Scalar, tanf, 0, eps));
            const f = 1 / tanf;

            return perspective(
                f / aspect,
                f * @intToFloat(Scalar, clip.y_dir),
                near,
                far,
                clip,
            );
        }

        /// Build a symmetrical perspective projection matrix given the bounds of the frustum
        pub fn perspectiveFrustum(
            l: Scalar, // left coordinate of the frustum
            r: Scalar, // right coordinate of the frustum
            b: Scalar, // bottom coordinate of the frustum
            t: Scalar, // top coordinate of the frustum
            near: Scalar, // distance of the near plane from the camera (along positive Z)
            far: Scalar, // distance of the far plane from the camera (along positive Z)
            clip: ClipSpace,
        ) Self {
            var w = r - l;
            var h = (t - b) * @intToFloat(Scalar, clip.y_dir);

            var mat = perspective(2 / w, 2 / h, near, far, clip);
            mat.elem[2][0] = (r + l) / w;
            mat.elem[2][1] = (t + b) / h;
            return mat;
        }

        fn perspective(
            x: Scalar,
            y: Scalar,
            near: Scalar,
            far: Scalar,
            clip: ClipSpace,
        ) Self {
            // See:
            // https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml
            // http://www.songho.ca/opengl/gl_projectionmatrix.html
            // https://vincent-p.github.io/posts/vulkan_perspective_matrix

            assert(near > 0 and far > 0 and far > near);
            assert(!math.approxEqAbs(Scalar, far, near, eps));

            const zn = @intToFloat(Scalar, clip.z_near);
            const zf = @intToFloat(Scalar, clip.z_far);
            const a = near * (zf - zn) / (near - far);
            const b = near * (zn + a);

            var mat = zero;

            mat.elem[0][0] = x;
            mat.elem[1][1] = y;
            mat.elem[2][2] = a;
            mat.elem[3][2] = b;
            mat.elem[2][3] = -1;

            return mat;
        }

        pub fn orthographic(
            w: Scalar,
            h: Scalar,
            near: Scalar,
            far: Scalar,
            clip: ClipSpace,
        ) Self {
            assert(!math.approxEqAbs(Scalar, w, 0, eps));
            assert(!math.approxEqAbs(Scalar, h, 0, eps));
            assert(!math.approxEqAbs(Scalar, far, near, eps));

            const zn = @intToFloat(Scalar, clip.z_near);
            const zf = @intToFloat(Scalar, clip.z_far);
            const z_scale = (zf - zn) / (near - far);

            var mat = zero;

            mat.elem[0][0] = 2 / w;
            mat.elem[1][1] = 2 / h * @intToFloat(Scalar, clip.y_dir);
            mat.elem[2][2] = z_scale;
            mat.elem[3][2] = zf - far * z_scale;
            mat.elem[3][3] = 1;

            return mat;
        }

        //=== Algebra ===//

        pub fn transpose(self: Self) Self {
            return fromRows(
                self.elem[0],
                self.elem[1],
                self.elem[2],
                self.elem[3],
            );
        }

        pub fn mul(self: Self, rh: anytype) mulType(@TypeOf(rh)) {
            const x: V = self.elem[0];
            const y: V = self.elem[1];
            const z: V = self.elem[2];
            const w: V = self.elem[3];

            switch (@TypeOf(rh)) {
                // Matrix-scalar multiplication
                Scalar, comptime_float, comptime_int => {
                    const n = @splat(4, @as(Scalar, rh));
                    return Self{ .elem = .{ x * n, y * n, z * n, w * n } };
                },
                // Matrix-matrix multiplication
                Self => {
                    var out: Self = undefined;

                    comptime var col = 0;
                    inline while (col < 4) : (col += 1) {
                        var out_col = x * @splat(4, rh.elem[col][0]);
                        out_col += y * @splat(4, rh.elem[col][1]);
                        out_col += z * @splat(4, rh.elem[col][2]);
                        out_col += w * @splat(4, rh.elem[col][3]);

                        out.elem[col] = out_col;
                    }

                    return out;
                },
                // Matrix-vector multiplication
                Vec2(Scalar, Tag) => {
                    return Vec2(Scalar, Tag).init(
                        self.elem[0][0] * rh.x + self.elem[1][0] * rh.y + self.elem[3][0],
                        self.elem[0][1] * rh.x + self.elem[1][1] * rh.y + self.elem[3][1],
                    );
                },
                Vec3(Scalar, Tag) => {
                    var out = x * @splat(4, rh.x);
                    out += y * @splat(4, rh.y);
                    out += z * @splat(4, rh.z);
                    out += w;
                    return Vec3(Scalar, Tag).init(out[0], out[1], out[2]);
                },
                Vec4(Scalar, Tag) => {
                    var out = x * @splat(4, rh.x);
                    out += y * @splat(4, rh.y);
                    out += z * @splat(4, rh.z);
                    out += w * @splat(4, rh.w);
                    return Vec4(Scalar, Tag).init(out[0], out[1], out[2], out[3]);
                },
                else => {
                    @compileError("Matrix multiplication not implemented for " ++
                        @typeName(@TypeOf(rh)));
                },
            }
        }

        fn mulType(comptime Rh: type) type {
            return switch (Rh) {
                Scalar, comptime_float, comptime_int => Self,
                else => Rh,
            };
        }
    };
}

//==== Utility ====//

pub const deg2rad = math.pi / 180.0;
pub const rad2deg = 180.0 / math.pi;

//==== Testing ====//

const expect = std.testing.expect;

test "Vec3" {
    const v = Vec3S.init(0, -8, 6);

    try expect(v.norm() == 10);
}

test "Mat4" {
    const i = Mat4S.identity;
    const m = i.mul(2);
    const n = m.mul(3);
    try expect(i.elem[1][1] == 1);
    try expect(m.elem[1][1] == 2);
    try expect(n.elem[1][1] == 6);

    const a = Mat4S.fromColumns(
        Vec4S.init(0.1, 0.5, 0.9, 1.3),
        Vec4S.init(0.2, 0.6, 1.0, 1.4),
        Vec4S.init(0.3, 0.7, 1.1, 1.5),
        Vec4S.init(0.4, 0.8, 1.2, 1.6),
    );

    const b = Mat4S.fromRows(
        Vec4S.init(1.7, 1.8, 1.9, 2.0),
        Vec4S.init(2.1, 2.2, 2.3, 2.4),
        Vec4S.init(2.5, 2.6, 2.7, 2.8),
        Vec4S.init(2.9, 3.0, 3.1, 3.2),
    );

    const c = a.mul(b);
    try expect(almostEq(c.elem[0], [4]f32{ 2.5, 6.18, 9.86, 13.54 }));
    try expect(almostEq(c.elem[1], [4]f32{ 2.6, 6.44, 10.28, 14.12 }));
    try expect(almostEq(c.elem[2], [4]f32{ 2.7, 6.7, 10.7, 14.7 }));
    try expect(almostEq(c.elem[3], [4]f32{ 2.8, 6.96, 11.12, 15.28 }));

    const d = Mat4S.fromColumns(
        [_]f32{ 1.7, 1.8, 1.9, 2.0 },
        [_]f32{ 2.1, 2.2, 2.3, 2.4 },
        [_]f32{ 2.5, 2.6, 2.7, 2.8 },
        [_]f32{ 2.9, 3.0, 3.1, 3.2 },
    );
    const t = b.transpose();
    comptime var col = 0;
    inline while (col < 4) : (col += 1) {
        try expect(t.elem[col][0] == b.elem[0][col]);
        try expect(t.elem[col][1] == b.elem[1][col]);
        try expect(t.elem[col][2] == b.elem[2][col]);
        try expect(t.elem[col][3] == b.elem[3][col]);

        try expect(t.elem[col][0] == d.elem[col][0]);
        try expect(t.elem[col][1] == d.elem[col][1]);
        try expect(t.elem[col][2] == d.elem[col][2]);
        try expect(t.elem[col][3] == d.elem[col][3]);
    }
}

test "Mat4-Vec multiplication" {
    const m = Mat4S.identity;
    const v = Vec4S.init(1, 2, 3, 4);
    const r = m.mul(v);

    try expect(v.x == r.x);
    try expect(v.y == r.y);
    try expect(v.z == r.z);
    try expect(v.w == r.w);

    const n = Mat4S.fromRows(
        Vec4S.init(1, 0, 0, 0),
        Vec4S.init(0, 1, 0, 0),
        Vec4S.init(0, 0, 1, 0),
        Vec4S.init(2, 3, 4, 1),
    );
    const nv = n.mul(Vec4S.init(1, 2, 3, 1));

    const z = n.transpose().mul(Vec4S.init(1, 2, 3, 1));
    try expect(almostEq([_]f32{ nv.x, nv.y, nv.z, nv.w }, [_]f32{ 1, 2, 3, 21 }));
    try expect(almostEq([_]f32{ z.x, z.y, z.z, z.w }, [_]f32{ 3, 5, 7, 1 }));
}

test "Mat4 translation" {
    const t1 = Mat4S.translation(1, 2, 3);
    const t2 = Mat4S.translation(4, 5, 6);

    var n = Mat4S.rotationZ(math.pi / 3.0);

    std.debug.print("before: {?}\n", .{n});

    n = n.mul(t1);
    std.debug.print("after t1: {?}\n", .{n});

    n = n.mul(t2);
    std.debug.print("after t2: {?}\n", .{n});

    var v = Vec3S.init(0.5, -7, 3);
    v = n.mul(v);

    std.debug.print("v: {?}\n", .{v});
}

test "Mat4 lookAt" {
    const m = Mat4S.lookAt(Vec3S.init(0, 0, 3), Vec3S.zero, Vec3S.y_axis);
    const n = Mat4S.translation(0, 0, -3);

    try expect(almostEq(m.elem[0], n.elem[0]));
    try expect(almostEq(m.elem[1], n.elem[1]));
    try expect(almostEq(m.elem[2], n.elem[2]));
    try expect(almostEq(m.elem[3], n.elem[3]));
}

fn almostEq(v0: anytype, v1: anytype) bool {
    const T = @TypeOf(v0, v1);
    const Info = @typeInfo(T).Array;

    comptime var i = 0;
    inline while (i < Info.len) : (i += 1) {
        if (!math.approxEqAbs(Info.child, v0[i], v1[i], eps)) {
            return false;
        }
    }
    return true;
}
