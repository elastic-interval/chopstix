pub mod approach;
pub mod brick;
pub mod executor;
pub mod face;
pub mod placement;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Spin {
    Left,
    Right,
}

impl Spin {
    pub fn opposite(self) -> Self {
        match self {
            Spin::Left => Spin::Right,
            Spin::Right => Spin::Left,
        }
    }
}
