use super::executor::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FabricName {
    Column3,
    Column6,
    Flagellum,
    OpenClaw,
    Triped,
    Mockup,
}

pub const ALL_FABRICS: &[FabricName] = &[
    FabricName::Column3,
    FabricName::Column6,
    FabricName::Flagellum,
    FabricName::OpenClaw,
    FabricName::Triped,
    FabricName::Mockup,
];

impl FabricName {
    pub fn label(self) -> &'static str {
        match self {
            FabricName::Column3 => "Col 3",
            FabricName::Column6 => "Col 6",
            FabricName::Flagellum => "Flagel",
            FabricName::OpenClaw => "Claw",
            FabricName::Triped => "Triped",
            FabricName::Mockup => "Mockup",
        }
    }

    pub fn program(self) -> BuildProgram {
        match self {
            FabricName::Column3 => BuildProgram {
                seed: SeedKind::SingleTwist,
                face_nodes: vec![face("forward", column(2).build())],
            },

            FabricName::Column6 => BuildProgram {
                seed: SeedKind::SingleTwist,
                face_nodes: vec![face("forward", column(5).build())],
            },

            // Flagellum: long whip-like column
            FabricName::Flagellum => BuildProgram {
                seed: SeedKind::SingleTwist,
                face_nodes: vec![
                    face("forward", column(20).shrink_by(5.0).build()),
                ],
            },

            // Mockup: short column (seed + 2)
            FabricName::Mockup => BuildProgram {
                seed: SeedKind::SingleTwist,
                face_nodes: vec![
                    face("forward", column(2).shrink_by(12.0).build()),
                ],
            },

            // Open Claw: omni hub with 3 legs of 4, prisms at ends
            FabricName::OpenClaw => BuildProgram {
                seed: SeedKind::Omni,
                face_nodes: vec![
                    face("OmniBotX", column(4)
                        .shrink_by(20.0)
                        .mark("End")
                        .prism(250.0)
                        .build()),
                    face("OmniBotY", column(4)
                        .shrink_by(20.0)
                        .mark("End")
                        .prism(250.0)
                        .build()),
                    face("OmniBotZ", column(4)
                        .shrink_by(20.0)
                        .mark("End")
                        .prism(250.0)
                        .build()),
                    face("OmniTop", prism(200.0)),
                    face("OmniBot", open()),
                ],
            },

            // Triped: omni hub with 3 legs of 8, prisms at ends
            FabricName::Triped => BuildProgram {
                seed: SeedKind::Omni,
                face_nodes: vec![
                    face("OmniBotX", column(8)
                        .shrink_by(10.0)
                        .mark("End")
                        .prism(100.0)
                        .build()),
                    face("OmniBotY", column(8)
                        .shrink_by(10.0)
                        .mark("End")
                        .prism(100.0)
                        .build()),
                    face("OmniBotZ", column(8)
                        .shrink_by(10.0)
                        .mark("End")
                        .prism(100.0)
                        .build()),
                    face("OmniTop", prism(100.0)),
                    face("OmniBot", open()),
                ],
            },
        }
    }
}
