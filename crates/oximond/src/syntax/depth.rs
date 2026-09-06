//! Tree depth of a parsed chunk, measured without recursion.
//!
//! The parser is an explicit state machine, but tools that walk the tree
//! recursively (the compiler, drop glue) need stack in proportion to its
//! depth. Left-associative chains such as `a.b.c...` or `1 + 1 + ...` are
//! unbounded in Lua, so their callers use this to size a stack first.

use crate::syntax::{Block, Chunk, Expr, Statement, TableField};

enum Node<'a> {
    Statement(&'a Statement),
    Expr(&'a Expr),
}

impl Chunk {
    /// Returns the deepest nesting of statements and expressions, counting
    /// each statement and each expression as one level.
    #[must_use]
    pub fn nesting_depth(&self) -> usize {
        let pending: Vec<(Node<'_>, usize)> = self
            .statements
            .iter()
            .map(|statement| (Node::Statement(statement), 1))
            .collect();
        nesting_depth(pending)
    }
}

impl Statement {
    /// Returns this statement's deepest nesting, counting the statement and
    /// each nested statement or expression as one level. The walk is iterative
    /// so compilers can check a borrowed statement before recursive lowering.
    #[must_use]
    pub fn nesting_depth(&self) -> usize {
        nesting_depth(vec![(Node::Statement(self), 1)])
    }
}

fn nesting_depth(mut pending: Vec<(Node<'_>, usize)>) -> usize {
    let mut deepest = 0;
    while let Some((node, depth)) = pending.pop() {
        deepest = deepest.max(depth);
        let child = depth + 1;
        match node {
            Node::Statement(statement) => push_statement(&mut pending, statement, child),
            Node::Expr(expression) => push_expression(&mut pending, expression, child),
        }
    }
    deepest
}

fn push_block<'a>(pending: &mut Vec<(Node<'a>, usize)>, block: &'a Block, depth: usize) {
    pending.extend(
        block
            .statements
            .iter()
            .map(|statement| (Node::Statement(statement), depth)),
    );
}

fn push_expressions<'a>(
    pending: &mut Vec<(Node<'a>, usize)>,
    expressions: &'a [Expr],
    depth: usize,
) {
    pending.extend(
        expressions
            .iter()
            .map(|expression| (Node::Expr(expression), depth)),
    );
}

fn push_statement<'a>(
    pending: &mut Vec<(Node<'a>, usize)>,
    statement: &'a Statement,
    depth: usize,
) {
    match statement {
        Statement::Do(body) => push_block(pending, body, depth),
        Statement::Local { values, .. }
        | Statement::Global { values, .. }
        | Statement::Return { values, .. } => push_expressions(pending, values, depth),
        Statement::Assign {
            targets, values, ..
        } => {
            push_expressions(pending, targets, depth);
            push_expressions(pending, values, depth);
        }
        Statement::Function { body, .. } | Statement::LocalFunction { body, .. } => {
            push_block(pending, body, depth);
        }
        Statement::If {
            condition,
            body,
            else_ifs,
            else_body,
            ..
        } => {
            pending.push((Node::Expr(condition), depth));
            push_block(pending, body, depth);
            for else_if in else_ifs {
                pending.push((Node::Expr(&else_if.condition), depth));
                push_block(pending, &else_if.body, depth);
            }
            if let Some(else_body) = else_body {
                push_block(pending, else_body, depth);
            }
        }
        Statement::While {
            condition, body, ..
        }
        | Statement::Repeat {
            body, condition, ..
        } => {
            pending.push((Node::Expr(condition), depth));
            push_block(pending, body, depth);
        }
        Statement::NumericFor {
            start,
            limit,
            step,
            body,
            ..
        } => {
            pending.push((Node::Expr(start), depth));
            pending.push((Node::Expr(limit), depth));
            if let Some(step) = step {
                pending.push((Node::Expr(step), depth));
            }
            push_block(pending, body, depth);
        }
        Statement::GenericFor {
            expressions, body, ..
        } => {
            push_expressions(pending, expressions, depth);
            push_block(pending, body, depth);
        }
        Statement::Call(expression) => pending.push((Node::Expr(expression), depth)),
        Statement::Break { .. } | Statement::Goto { .. } | Statement::Label { .. } => {}
    }
}

fn push_expression<'a>(pending: &mut Vec<(Node<'a>, usize)>, expression: &'a Expr, depth: usize) {
    match expression {
        Expr::Name(_)
        | Expr::Nil(_)
        | Expr::Boolean(..)
        | Expr::Integer(..)
        | Expr::IntegerBoundary(_)
        | Expr::Number(..)
        | Expr::Int64(..)
        | Expr::UInt64(..)
        | Expr::String(..)
        | Expr::Vararg(_) => {}
        Expr::Table { fields, .. } => {
            for field in fields {
                match field {
                    TableField::List(value) | TableField::Name { value, .. } => {
                        pending.push((Node::Expr(value), depth));
                    }
                    TableField::Key { key, value } => {
                        pending.push((Node::Expr(key), depth));
                        pending.push((Node::Expr(value), depth));
                    }
                }
            }
        }
        Expr::Function { body, .. } => push_block(pending, body, depth),
        Expr::Index { table, key, .. } => {
            pending.push((Node::Expr(table), depth));
            pending.push((Node::Expr(key), depth));
        }
        Expr::Binary { left, right, .. } => {
            pending.push((Node::Expr(left), depth));
            pending.push((Node::Expr(right), depth));
        }
        Expr::Unary { expression, .. } | Expr::Parenthesized { expression, .. } => {
            pending.push((Node::Expr(expression), depth));
        }
        Expr::Call {
            callee, arguments, ..
        } => {
            pending.push((Node::Expr(callee), depth));
            push_expressions(pending, arguments, depth);
        }
        Expr::MethodCall {
            receiver,
            arguments,
            ..
        } => {
            pending.push((Node::Expr(receiver), depth));
            push_expressions(pending, arguments, depth);
        }
    }
}
