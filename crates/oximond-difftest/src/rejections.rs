//! Small invalid Lua chunks for compiler acceptance comparisons.

use crate::profile::Profile;

/// Each source must fail to compile in the stock interpreter for `profile`.
/// The runner checks that premise before comparing oximond's acceptance.
#[must_use]
pub fn rejections(profile: Profile) -> Vec<(&'static str, &'static str)> {
    let dialect = profile.features();
    let mut cases = vec![("missing_end", "do local value = 1")];
    if dialect.goto {
        cases.push(("unknown_label", "goto missing_label"));
        cases.push((
            "jump_into_local_scope",
            "goto after_local; local value = 1; ::after_local:: print(value)",
        ));
    }
    if dialect.close_and_const {
        cases.push(("const_local_write", "local value <const> = 1; value = 2"));
    }
    if dialect.lua55 {
        cases.push((
            "const_global_write",
            "global<const> rejected_global = 1; rejected_global = 2",
        ));
        cases.push((
            "named_vararg_write",
            "local function f(... extras) extras = {} end",
        ));
    }
    cases
}

#[cfg(test)]
mod tests {
    use super::rejections;
    use crate::profile::Profile;

    #[test]
    fn dialect_specific_rejections_are_only_selected_where_syntax_exists() {
        for profile in Profile::ALL {
            let names: Vec<_> = rejections(profile)
                .into_iter()
                .map(|(name, _)| name)
                .collect();
            let dialect = profile.features();
            assert!(names.contains(&"missing_end"));
            assert_eq!(names.contains(&"unknown_label"), dialect.goto);
            assert_eq!(
                names.contains(&"const_local_write"),
                dialect.close_and_const
            );
            assert_eq!(names.contains(&"const_global_write"), dialect.lua55);
            assert_eq!(names.contains(&"named_vararg_write"), dialect.lua55);
        }
    }
}
