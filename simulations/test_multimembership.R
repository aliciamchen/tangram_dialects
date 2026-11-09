# Tests for the shared multiple-membership speaker intercept
# (simulations/multimembership.R): Rscript simulations/test_multimembership.R
#
# The design-matched simulation study that checks convergence, ordering
# invariance, and interval calibration at scale is
# simulations/simulate_mm_speaker.R; these are the fast structural checks.

if (!exists("lmer_multimember")) {
  source(here::here("stats", "config.R"))
  source(here::here("simulations", "multimembership.R"))
}
if (!exists("check")) {
  source(here::here("stats", "tests", "check.R"))
}

set.seed(4)

# ── membership_weights / speaker_pair_weights ───────────────────────────────

W <- speaker_pair_weights(
  c("a", "a", "b"),
  c("b", "c", "c"),
  levels = c("a", "b", "c")
)
check(
  "a pair's weights are 1/2 on each speaker and 0 elsewhere",
  all(dim(W) == c(3, 3)) &&
    all(Matrix::colSums(W) == 1) &&
    all(W@x == 0.5) &&
    length(W@x) == 6
)
check(
  "weights are indexed by the level order given, not by appearance",
  identical(rownames(W), c("a", "b", "c")) &&
    all(as.numeric(W[, 2]) == c(0.5, 0, 0.5))
)
check(
  "a speaker paired with themselves is rejected",
  inherits(try(speaker_pair_weights("a", "a"), silent = TRUE), "try-error")
)
check(
  "a member missing from the level set is rejected",
  inherits(
    try(speaker_pair_weights("a", "z", levels = c("a", "b")), silent = TRUE),
    "try-error"
  )
)

# ── The modular fit reproduces lme4 when every row has one member ───────────
#
# This is the check that the Zt surgery is right: with one member per
# observation at weight 1 the term is an ordinary scalar random intercept, so
# the fit has to match lmer() exactly.

n <- 300
one <- tibble(
  target = factor(sample(paste0("t", 1:8), n, TRUE)),
  speaker = factor(sample(paste0("s", 1:9), n, TRUE)),
  x = rnorm(n)
)
one$y <- 1 +
  0.4 * one$x +
  rnorm(9)[as.integer(one$speaker)] +
  0.5 * rnorm(8)[as.integer(one$target)] +
  rnorm(n, sd = 0.7)

ref <- lmer(
  y ~ x + (1 | target) + (1 | speaker),
  data = one,
  control = lmerControl(optimizer = "bobyqa")
)
single <- lmer_multimember(
  y ~ x + (1 | target) + (1 | speaker),
  data = one[, c("y", "x", "target")],
  memberships = list(
    speaker = membership_weights(
      as.list(as.character(one$speaker)),
      levels(one$speaker)
    )
  )
)
check(
  "single membership reproduces lmer's fixed effects and standard errors",
  max(abs(fixef(ref) - fixef(single))) < 1e-6 &&
    max(abs(coef(summary(ref))[, 2] - coef(summary(single))[, 2])) < 1e-5
)
check(
  "single membership reproduces lmer's REML criterion and variances",
  abs(REMLcrit(ref) - REMLcrit(single)) < 1e-6 &&
    abs(sigma(ref) - sigma(single)) < 1e-6
)
check(
  "random effects come back indexed by speaker",
  setequal(rownames(ranef(single)$speaker), levels(one$speaker))
)

# ── A pairwise design: one effect per speaker, shared across positions ──────

speakers <- paste0("s", 1:9)
groups <- setNames(rep(c("A", "B", "C"), each = 3), speakers)
pairs <- as.data.frame(t(combn(speakers, 2)), stringsAsFactors = FALSE)
names(pairs) <- c("speaker1", "speaker2")
targets <- paste0("t", 1:8)
pw <- do.call(
  rbind,
  lapply(targets, function(tg) {
    transform(pairs, target = tg)
  })
)
pw$sameGroup <- as.numeric(groups[pw$speaker1] == groups[pw$speaker2])
pw$gameId <- "g1"

speaker_effects <- setNames(rnorm(9, sd = 0.30), speakers)
target_effects <- setNames(rnorm(8, sd = 0.20), targets)
pw$similarity <- 0.30 +
  0.25 * pw$sameGroup +
  0.5 * (speaker_effects[pw$speaker1] + speaker_effects[pw$speaker2]) +
  target_effects[pw$target] +
  rnorm(nrow(pw), sd = 0.15)

fit_pw <- function(d) {
  lmer_multimember(
    similarity ~ sameGroup + (1 | target) + (1 | speaker),
    data = d,
    memberships = list(
      speaker = speaker_pair_weights(d$speaker1, d$speaker2, levels = speakers)
    )
  )
}
m_pw <- fit_pw(pw)

check(
  "one random effect per speaker, not one per pair position",
  nrow(ranef(m_pw)$speaker) == length(speakers) &&
    setequal(rownames(ranef(m_pw)$speaker), speakers)
)
check(
  "the same_group effect is recovered on a design-matched sample",
  abs(fixef(m_pw)[["sameGroup"]] - 0.25) < 0.1
)

# Swapping the two positions is a relabelling of an unordered pair, so it must
# change nothing. The separate-intercepts model this replaces has no such
# guarantee, which is one reason it was replaced.
swapped <- pw
flip <- c(TRUE, FALSE)[sample(2, nrow(pw), TRUE)]
swapped$speaker1[flip] <- pw$speaker2[flip]
swapped$speaker2[flip] <- pw$speaker1[flip]
m_swapped <- fit_pw(swapped)

check(
  "the estimate is invariant to which speaker is listed first",
  abs(fixef(m_pw)[["sameGroup"]] - fixef(m_swapped)[["sameGroup"]]) < 1e-8
)
check(
  "the standard error is invariant to pair ordering",
  abs(
    coef(summary(m_pw))["sameGroup", "Std. Error"] -
      coef(summary(m_swapped))["sameGroup", "Std. Error"]
  ) <
    1e-8
)
check(
  "the speaker variance is invariant to pair ordering",
  abs(sigma(m_pw) - sigma(m_swapped)) < 1e-8 &&
    max(abs(getME(m_pw, "theta") - getME(m_swapped, "theta"))) < 1e-6
)

# The separate-intercepts model is not invariant, so its estimates
# depend on an arbitrary ordering. Kept as a test so the difference between
# the two structures stays visible.
sep <- function(d) {
  lmer(
    similarity ~ sameGroup + (1 | target) + (1 | speaker1) + (1 | speaker2),
    data = d,
    control = lmerControl(optimizer = "bobyqa")
  )
}
check(
  "the separate-intercepts model it replaces is not ordering invariant",
  abs(fixef(sep(pw))[["sameGroup"]] - fixef(sep(swapped))[["sameGroup"]]) > 1e-8
)
