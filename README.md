# Gambit: The Robot Chess Teacher

> **Status:** Proof of concept, in active development (Mark-I). Not yet a finished chess player.
>
> **Docs:** a Wiki covering every design decision and code walkthrough is planned. See [Documentation and design decisions](#documentation-and-design-decisions).

Gambit is an SO-ARM101 robot arm that plays physical chess against a human and adapts to that player's habits in order to teach them. The long-term goal is a platform that teaches complex board games by playing them, not by lecturing about them.

<!-- TODO: add demo video or GIF here -->

---

## Why Gambit

Most chess platforms adjust a player's rating from game results alone. A player who has been away from the game keeps their chess intuition, but their habits get sloppier and their pattern recognition gets weaker. Losing games because of that rust says little about their real understanding.

Gambit takes a different approach: it estimates a player's current level from their **habits** (recurring mistakes, missed tactics, risky or passive tendencies) and chooses moves that target those specific weaknesses. The name nods to the chess gambit, an opening that trades something early for a long-term advantage, and is also a fan tribute to Gambit, the Marvel Comics character.

### Constructive smurfing

"Smurfing" normally means a strong player hiding behind a low rating to beat weaker opponents. Gambit uses the idea in reverse, for the player's benefit. It deliberately plays below its full strength, aims its moves at the player's weak points, and only reveals more of its real ability as the player improves.

Design goals:

- Keep games close, so players stay engaged instead of discouraged.
- Punish habits, not playstyles. Safe, aggressive, and passionate styles are all valid. Repeated mistakes are what lose games.
- Build confidence in decision making, pattern recognition, and handling pressure.

Open design question: should Gambit tell players it is adjusting its strength, and when? Decide before any user testing.

---

## How Gambit addresses these problems

This table describes the planned approach. Nothing here is validated yet, and each row will get its own design decision page in the Wiki.

| Problem | How Gambit addresses it | Status |
|---|---|---|
| Ratings reflect results, not habits | Analyze each of the player's moves with an engine (move quality, missed tactics, recurring patterns) to estimate habits and current level | Planned |
| Returning players are punished for rust | Match strength to the player's current habits instead of an old rating, and keep games close | Planned |
| The player model can be wrong | Track confidence, fall back to solid strength-limited play when unsure, and test against players who differ from their history | Planned |
| A budget arm is imprecise | Measure repeatability first, then use camera correction, grasp verification, and automatic retry. Compare scripted motion with a learned policy | Planned |
| The engine's board and the physical board can disagree | The camera is the source of truth. Infer the human's move from the changed squares and the legal moves, and ask the human when unsure | Planned |
| Learning by doing beats lecturing | Gambit plays and physically demonstrates corrections instead of explaining theory | Later phase |
| Hidden strength changes may feel patronizing | Decide whether and when to disclose before user testing | Open question |

---

## Scope

### Mark-I (current)

- Standard pick-and-place actions on a physical chessboard with an SO-ARM101.
- Camera-based board and piece recognition.
- A chess engine that plays at a chosen strength and adapts to the player where its player model is confident. Planned: when the model is uncertain, fall back to solid, strength-limited play.

### Later phases

- **Shogi.** A bigger jump for the arm than for the engine: pieces are flat wedges of one color, so vision must read orientation, and captured pieces return to play, so the arm needs a piece stand and must place pieces onto the board ("drops").
- **Other complex board games.**
- **Demonstration feedback:** physically replaying positions to show players how to correct a habit.
- **Optional voice interaction.** Not committed. The project favors learning by doing over talking.
- **Custom-built robot.** Possible replacement of the SO-ARM101 with a more rigid, more repeatable arm if measurements show the budget arm cannot meet the accuracy targets.

---

## Milestones

Targets marked *(proposed)* are placeholders to confirm or change after measurements exist.

### Hardware and execution

- [ ] Arm assembled, servo identifiers assigned, calibrated, and teleoperating
- [ ] Baseline repeatability measured (command the same pose 100 times, record the spread in millimeters)
- [ ] Board position registered with fixed markers at the board corners
- [ ] Rigid mounting of arm and board on one plate
- [ ] Gripper fingers shaped to the chess pieces, with grasp verification
- [ ] Pick-and-place success on the first attempt: **55 to 60 percent** (initial target)
- [ ] Pick-and-place success after automatic verify-and-retry: 95 percent *(proposed)*
- [ ] Per-move success of 99 percent or better *(stretch)*
- [ ] Complete a full game with no human intervention *(proposed: 10 consecutive games)*

### Perception and rules

- [ ] Identify the board state from the camera
- [ ] Verify physical placement: piece centered on its square within tolerance
- [ ] Verify rule legality: the move is legal chess (checked separately from placement)
- [ ] Detect mismatches between the physical board and the engine's expected board, and recover

### Teaching

- [ ] Estimate player habits from game history
- [ ] Fallback behavior: when the player model is uncertain, or the player departs sharply from their history, play solid strength-limited moves instead of exploiting a habit
- [ ] Out-of-distribution testing: measure how the player model behaves against stand-in players whose style differs from their history, and against players much stronger or weaker than estimated
- [ ] Rating-targeted play at around 1700 Elo *(current target)*
- [ ] Balanced outcomes: define the metric (for example, the share of games drawn or decided narrowly) and a target over a fixed number of games. Initial aspiration: 95 to 99 percent for typical play at the estimated rating (an aspiration, not a guarantee)
- [ ] Stretch: adapt to players near Grandmaster level (2500 Elo and above)

---

## Hardware

| Component | Status |
|---|---|
| SO-ARM101 leader and follower arms | In hand |
| Waveshare Bus Servo Adapter (A) | In hand, one per arm |
| Camera (overhead, rigidly mounted) | TBD |
| Lighting (diffuse, constant) | TBD |
| Chessboard and pieces (weighted, uniform) | TBD |
| Laptop or Jetson Orin for training and inference | TBD |
| STM32 board for the real-time layer | Optional, planned |

Notes:

- The adapter's jumper selects the control mode: position A for UART, position B for USB.
- Servo power must match the servo voltage. Check your kit's servo rating before choosing a supply.

---

## Software

| Layer | Plan |
|---|---|
| Arm control and teleoperation | LeRobot on a laptop first |
| Training | Imitation learning on a laptop or Jetson Orin graphics processor |
| Chess engine | TBD (check its license, see below) |
| Vision | TBD |
| Real-time layer | Optional bare-metal STM32 firmware that owns the servo bus, enforces joint and torque limits, and talks to the computer over a serial or network link |

Development order: get the arm working and trained on the laptop first, then add the STM32 firmware as a real-time layer. Measure both paths on the same task.

---

## Known limitations and risks

- The SO-ARM101 is a low-cost arm. Expect backlash, limited repeatability, and calibration drift.
- Chess pieces are small and close together, so a small placement error can knock a neighbor.
- The engine's board state can diverge from the physical board if a piece lands badly or falls. The camera must be the source of truth after every move.
- Special moves need their own sequences: captures, castling, en passant, and promotion.

---

## What Gambit does and does not claim

- Gambit does not claim to beat every player or to avoid losing. It is designed to keep games close, so wins and losses both happen by design.
- The chess engine searches any legal position, so unusual or off-book moves do not break it. What can fail is the **player model**, which estimates a player's habits and level from past games. It can be wrong when a player behaves unlike their history: new openings, deliberate off-book play, a player much stronger or weaker than estimated, or a player hiding their strength.
- Planned: when the player model is uncertain, Gambit plays solid, strength-limited moves at its last estimated level and does not try to exploit a habit.
- The balanced-outcomes figure in the milestones is an aspiration for typical play, not a guarantee against every opponent.

---

## Documentation and design decisions

A project Wiki is planned. It will explain why Gambit is built the way it is and how each piece of code works, from Mark-I through to the final version.

<!-- TODO: add the Wiki link once it exists -->

Planned Wiki contents:

- **Design decision pages:** one per decision, covering the problem, the options considered, the choice made, and the consequences. Decisions that change are marked as superseded, not deleted.
- **Code walkthroughs:** for each version, how each module is written and why (arm control, perception, chess engine and player model, game manager, STM32 firmware).
- **Measurements and test logs:** repeatability, placement success rates, calibration results, and game statistics.
- **Glossary:** the chess, robotics, and rating terms used in the project.

| Version | Scope | Status |
|---|---|---|
| Mark-I | Chess on the SO-ARM101, proof of concept | In progress |
| Later versions | To be defined as Mark-I measurements come in | Not started |
| Final version | Custom-built robot for chess, Shogi, and other complex board games | Not started |

Until the Wiki exists, this README is the source of truth.

---

## Project log

| Date | Update |
|---|---|
| 2026-10 | Project started. Arm identified, hardware and licensing researched. |
| 2026-10-13 | README expanded with the problem and approach table, claims and limits, and the documentation plan. |

---

## Open questions

- What placement tolerance does a chess square actually require, and does the SO-ARM101 meet it?
- Which board size fits the arm's reach?
- How will player habits be measured, and from how many games?
- Should the robot disclose that it is adjusting its strength?
- What should Gambit do when the player's play departs sharply from their history?
- Should Gambit ever play at full strength to avoid losing, or are losses an accepted part of teaching?
- What are the minimum and maximum strengths Gambit will play at?

---

## License

License to be decided. Before choosing, check the license of every dependency (chess engine, chess libraries, LeRobot, arm designs, vendor servo code) and confirm they are compatible with your plans for the project.

---

## Acknowledgments

- SO-ARM101 open hardware design and LeRobot
- Waveshare Bus Servo Adapter documentation

Gambit is a character owned by Marvel. This project is an independent fan tribute and is not affiliated with or endorsed by Marvel.
