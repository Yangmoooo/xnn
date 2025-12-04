const std = @import("std");

pub fn build(b: *std.Build) void {
    // 构建目标
    const target = b.standardTargetOptions(.{});

    // 构建优化模式
    const optimize = b.standardOptimizeOption(.{});

    // 添加一个二进制可执行程序构建
    const exe = b.addExecutable(.{
        .name = "main",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
        }),
    });

    // 添加 C 源代码文件
    // 单文件可以使用 addCSourceFile
    // 参数为 .file = b.path("xxx.c") 和 flags
    exe.addCSourceFiles(.{
        .files = &.{ "main.c", "dataset.c", "model.c", "utils.c" },
        .flags = &.{
            "-march=native",
            "-ffast-math",
            "-Wall",
            "-Wextra",
            "-pedantic",
        },
    });

    // 添加头文件路径
    exe.addIncludePath(b.path("include"));

    // 链接 C 标准库
    // 同理对于 C++ 标准库可以使用 linkLibCpp
    exe.linkLibC();

    // 链接系统库 ncurses
    // exe.linkSystemLibrary("ncurses");

    // 添加到顶级 install step 中作为依赖
    b.installArtifact(exe);

    // 创建一个运行
    const run_cmd = b.addRunArtifact(exe);

    // 依赖于构建
    run_cmd.step.dependOn(b.getInstallStep());

    // 运行时参数传递
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // 运行的 step
    const run_step = b.step("run", "Run the app");
    // 依赖于前面的运行
    run_step.dependOn(&run_cmd.step);
}
