use std::{borrow::Cow, env::Args, num::NonZero};

use crate::Config;

pub(crate) fn parse_args(args: &mut Args) -> Config {
    let iter = args.by_ref();
    let mut config = Config::default();

    // Skip first arg, which is the binary name
    if iter.next().is_none() {
        return config;
    }

    // // Then '--' (at least when invoked from cargo bench)
    // if iter.next().is_none() {
    //     return config;
    // }

    loop {
        let Some(arg) = iter.next() else {
            break;
        };

        if arg == "--" {
            break;
        }

        if arg == "--bench" {
            continue;
        }

        if let Some(long_arg) = arg.strip_prefix("--") {
            let (key_str, value_str) = if long_arg.contains("=") {
                // The arg is key=value
                let mut split = long_arg.splitn(2, '=');
                (
                    Cow::Borrowed(split.next().unwrap()),
                    Cow::Borrowed(split.next().unwrap()),
                )
            } else {
                // The value is on the next arg
                let key_str = long_arg;
                let value_str = iter.next().unwrap();
                (Cow::Borrowed(key_str), Cow::Owned(value_str))
            };

            if key_str == "num-samples" {
                let value = value_str.parse::<usize>().unwrap();
                config.bencher_config.num_samples =
                    NonZero::new(value).expect("num-samples must be > 0");
            } else if key_str == "iter-per-sample" {
                let value = value_str.parse::<usize>().unwrap();
                config.bencher_config.iter_per_sample =
                    NonZero::new(value).expect("iter-per-sample must be > 0");
            } else if key_str == "warmup-samples" {
                config.bencher_config.warmup_samples = value_str.parse::<usize>().unwrap();
            } else {
                println!("Unknown argument: {arg}");
            }
        } else {
            config.filter = Some(regex::Regex::new(&arg).unwrap());
        }
    }

    config
}
