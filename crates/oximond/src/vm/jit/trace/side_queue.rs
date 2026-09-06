//! The queue of recorded sides that wait for their tree's compile.
//!
//! Caches of what the queue plans to (`InstalledTrace::seed_plan`) are keyed
//! on its generation. The generation advances in every method that changes
//! the queue, and the items are private, so no change can forget to advance
//! it.

/// Queued entries of type `E` and the generation of the queue.
#[derive(Debug)]
pub(super) struct SideQueue<E> {
    items: Vec<E>,
    generation: u64,
}

impl<E> Default for SideQueue<E> {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            generation: 0,
        }
    }
}

impl<E> SideQueue<E> {
    /// The queued entries, oldest first.
    pub(super) fn queued(&self) -> &[E] {
        &self.items
    }

    /// Advances whenever the queue changes.
    pub(super) fn generation(&self) -> u64 {
        self.generation
    }

    /// Adds `entry` at the back.
    pub(super) fn queue_push(&mut self, entry: E) {
        self.generation += 1;
        self.items.push(entry);
    }

    /// Adds `entry` at the front.
    pub(super) fn queue_insert_front(&mut self, entry: E) {
        self.generation += 1;
        self.items.insert(0, entry);
    }

    /// Removes and returns every entry.
    pub(super) fn queue_take(&mut self) -> Vec<E> {
        self.generation += 1;
        std::mem::take(&mut self.items)
    }

    /// Removes and returns the first entry.
    ///
    /// # Panics
    ///
    /// If the queue is empty.
    pub(super) fn queue_remove_first(&mut self) -> E {
        self.generation += 1;
        self.items.remove(0)
    }

    /// Adds `entries` at the front, in order.
    pub(super) fn queue_splice_front(&mut self, entries: impl IntoIterator<Item = E>) {
        self.generation += 1;
        self.items.splice(0..0, entries);
    }
}

#[cfg(test)]
mod tests {
    use super::SideQueue;

    #[test]
    fn every_change_of_the_queue_advances_its_generation() {
        let mut queue = SideQueue::<u32>::default();
        let mut last = queue.generation();
        let mut advanced = |queue: &SideQueue<u32>| {
            let now = queue.generation();
            assert!(now > last);
            last = now;
        };
        queue.queue_push(1);
        advanced(&queue);
        queue.queue_insert_front(0);
        advanced(&queue);
        queue.queue_splice_front([7, 8]);
        advanced(&queue);
        assert_eq!(queue.queued(), &[7, 8, 0, 1]);
        assert_eq!(queue.queue_remove_first(), 7);
        advanced(&queue);
        assert_eq!(queue.queue_take(), vec![8, 0, 1]);
        advanced(&queue);
        assert_eq!(queue.queued(), []);
    }
}
