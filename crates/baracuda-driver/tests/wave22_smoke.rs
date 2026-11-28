// SPDX-License-Identifier: MIT OR Apache-2.0
//! GPU-gated integration test for Wave-22 Driver-API additions:
//! user objects attached to graphs.

#![cfg(feature = "device-tests")]

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use baracuda_driver::user_object::UserObject;
use baracuda_driver::{Context, Device, Graph, require};

static DESTRUCTOR_CALLS: AtomicU32 = AtomicU32::new(0);

#[test]
fn user_object_destructor_fires_when_graph_drops() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();

    DESTRUCTOR_CALLS.store(0, Ordering::SeqCst);

    // Capture a closure; the closure dropping implies destructor ran.
    let counter = Arc::new(());
    let _counter_clone = counter.clone();
    let uo = require!(
        UserObject::new(move || {
            DESTRUCTOR_CALLS.fetch_add(1, Ordering::SeqCst);
            drop(_counter_clone);
        }),
        "cuUserObjectCreate (CUDA 12+)"
    );

    let graph = Graph::new(&ctx).unwrap();
    // The graph takes its own reference; then give ours back (`release` consumes the value: nothing is left to release twice).
    graph.retain_user_object(&uo, 1).unwrap();
    uo.release().unwrap();

    assert_eq!(
        DESTRUCTOR_CALLS.load(Ordering::SeqCst),
        0,
        "destructor should not fire while graph holds ref"
    );

    // Drop graph -> releases user object -> destructor runs.
    drop(graph);

    // The destructor runs synchronously on graph destroy.
    assert_eq!(
        DESTRUCTOR_CALLS.load(Ordering::SeqCst),
        1,
        "destructor should have fired on graph drop"
    );
    assert_eq!(Arc::strong_count(&counter), 1, "Arc clone was dropped");

    drop(ctx);
}

/// One owning value per reference: a duplicate keeps the object alive after the first value is released, the destructor runs exactly once
/// when the last value goes, and neither `release` nor `Drop` gives a reference back that the value no longer owns.
#[test]
fn duplicate_keeps_the_object_alive_until_the_last_value_is_gone() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let _ctx = Context::new(&device).unwrap();

    let calls = Arc::new(AtomicU32::new(0));
    let c = calls.clone();
    let first = require!(
        UserObject::new(move || {
            c.fetch_add(1, Ordering::SeqCst);
        }),
        "cuUserObjectCreate (CUDA 12+)"
    );
    let second = first.duplicate().unwrap();

    // Giving back the first reference leaves the second: the destructor has not run.
    first.release().unwrap();
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "one reference is still owned"
    );

    // The last owning value going away (here by `Drop`) runs the destructor exactly once.
    drop(second);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "destructor runs once, on the last reference"
    );
}

/// A value released with `release` and one dropped without it each give back exactly one reference: with a graph holding one more, the
/// destructor runs when the graph goes, once, whichever way the value ended.
#[test]
fn release_and_drop_each_give_back_one_reference() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();

    for released in [true, false] {
        let calls = Arc::new(AtomicU32::new(0));
        let c = calls.clone();
        let uo = require!(
            UserObject::new(move || {
                c.fetch_add(1, Ordering::SeqCst);
            }),
            "cuUserObjectCreate (CUDA 12+)"
        );
        let graph = Graph::new(&ctx).unwrap();
        graph.retain_user_object(&uo, 1).unwrap();
        if released {
            uo.release().unwrap();
        } else {
            drop(uo);
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "the graph still holds a reference (released={released})"
        );
        drop(graph);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "destructor ran exactly once (released={released})"
        );
    }
}
