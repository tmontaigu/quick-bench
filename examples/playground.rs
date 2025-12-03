use std::hint::black_box;

use quick_bench::{Benchable, Bencher, Metric, Runner, WallTimeAnd, cpu_time::CpuLoad};

fn fibonacci_recursive_if(n: i32) -> i32 {
    if n <= 0 {
        0
    } else if n == 1 {
        1
    } else {
        fibonacci_recursive_if(n - 1) + fibonacci_recursive_if(n - 2)
    }
}
fn work() {
    let mut d = 0;
    for n in 0..20_000 {
        for m in 0..20_000 {
            d = black_box(d * n * m);
        }
    }
}

pub struct FibRecBench;

impl Benchable<i32, WallTimeAnd<CpuLoad>> for FibRecBench {
    fn write_name(&self, args: &i32, name: &mut String) {
        use std::fmt::Write;
        write!(name, "fibonnacci({args})").unwrap()
    }

    fn execute(&mut self, bencher: &mut Bencher<WallTimeAnd<CpuLoad>>, args: i32) {
        let _stats = bencher.bench(|| fibonacci_recursive_if(args));
    }
}

fn main() {
    let mut runner = Runner::<WallTimeAnd<CpuLoad>>::new().setup_from_cmdline();

    runner.run_with(FibRecBench, [10]);

    runner.run_with(
        ("hello", |bencher: &mut Bencher<WallTimeAnd<CpuLoad>>, _| {
            let stats = bencher.bench(|| fibonacci_recursive_if(10));
            println!("Yo: {}", stats.extra.mean_percent);
        }),
        [()],
    );

    runner.run("hello", |bencher| {
        let stats = bencher.bench(|| fibonacci_recursive_if(10));
        println!("Yo: {}", stats.extra.mean_percent);
    });

    runner.run_with(
        ("fibonnacci", |bencher: &mut Bencher<_>, n| {
            bencher.bench(|| fibonacci_recursive_if(n));
        }),
        [10, 20, 30, 32],
    );

    runner.run("parallel func", |bencher| {
        bencher.bench(|| {
            rayon::broadcast(|_| work());
        });
    });
}
