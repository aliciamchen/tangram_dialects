# Every analysis script states its inputs once, and the Makefile agrees.
#
# A script names the tables it reads in load_tables(...) and the estimates it
# builds on in require_estimates(), read_estimates() or load_fit(). The
# Makefile lists, per estimates table, the files that make it out of date.
# These tests read both and require them to be the same set, so that a script
# that starts reading another table cannot leave `make estimates` unaware of
# it. They also require every table a script loads to be used in it.

if (!exists("table_dependencies")) {
  source(here::here("stats", "config.R"))
}
if (!exists("check")) {
  source(here::here("stats", "tests", "check.R"))
}

root <- here::here()
relative <- function(paths) {
  sub(
    paste0("^", gsub("([.\\\\+*?\\[^$(){}|])", "\\\\\\1", root), "/"),
    "",
    paths
  )
}

scripts <- sort(unlist(lapply(
  c("checks", "primary", "secondary", "exploratory"),
  function(family) {
    list.files(
      file.path(root, "stats", family),
      pattern = "\\.R$",
      full.names = TRUE
    )
  }
)))
# stats/primary/h4.R is "primary/h4", the name of its estimates table
script_name <- function(path) {
  sub("\\.R$", "", sub("^stats/", "", relative(path)))
}
estimates_file <- function(name) {
  if (length(name) == 0) {
    return(character(0))
  }
  relative(file.path(estimates_dir, paste0(name, ".csv")))
}

# Every call in a script, at any depth
calls_in <- function(expr) {
  if (is.call(expr)) {
    c(list(expr), unlist(lapply(as.list(expr), calls_in), recursive = FALSE))
  } else if (is.expression(expr) || is.pairlist(expr)) {
    unlist(lapply(as.list(expr), calls_in), recursive = FALSE)
  } else {
    list()
  }
}
called <- function(calls, fn) {
  Filter(function(cl) is.name(cl[[1]]) && as.character(cl[[1]]) == fn, calls)
}
literal <- function(call, position, script, fn) {
  args <- as.list(call)[-1]
  if (length(args) < position || !is.character(args[[position]])) {
    stop(
      sprintf(
        "%s: argument %d of %s() must be a quoted name",
        script,
        position,
        fn
      ),
      call. = FALSE
    )
  }
  args[[position]]
}

# The script that writes an estimates table: a script's own table carries its
# name, and its further tables add a suffix (primary/h4_social_components)
names_of_scripts <- vapply(scripts, script_name, character(1))
writer_of <- function(estimates) {
  owners <- names_of_scripts[
    estimates == names_of_scripts |
      startsWith(estimates, paste0(names_of_scripts, "_"))
  ]
  if (length(owners) == 0) {
    stop("no script writes the estimates table ", estimates, call. = FALSE)
  }
  owners[which.max(nchar(owners))]
}

declared_inputs <- function(path) {
  script <- script_name(path)
  expr <- parse(path)
  calls <- calls_in(expr)
  tables <- unlist(lapply(called(calls, "load_tables"), function(cl) {
    vapply(
      seq_along(as.list(cl)[-1]),
      function(i) literal(cl, i, script, "load_tables"),
      character(1)
    )
  }))
  estimates <- c(
    vapply(
      called(calls, "require_estimates"),
      function(cl) literal(cl, 1, script, "require_estimates"),
      character(1)
    ),
    vapply(
      called(calls, "read_estimates"),
      function(cl) literal(cl, 1, script, "read_estimates"),
      character(1)
    )
  )
  fit_writers <- vapply(
    called(calls, "load_fit"),
    function(cl) literal(cl, 2, script, "load_fit"),
    character(1)
  )
  writers <- unique(c(
    vapply(estimates, writer_of, character(1)),
    sub("\\.R$", "", sub("^stats/", "", fit_writers))
  ))
  list(
    tables = tables,
    symbols = all.names(expr, functions = FALSE),
    files = unique(c(
      relative(unlist(lapply(tables, table_dependencies))),
      estimates_file(setdiff(writers, script))
    ))
  )
}

# The prerequisites Make holds for each estimates table, from its database
# (the pattern rule adds the script and the shared R code to every table)
make_database <- suppressWarnings(system2(
  "make",
  c(
    "-pq",
    "-C",
    shQuote(root),
    paste0("DATASET=", DATASET),
    "no-such-target"
  ),
  stdout = TRUE,
  stderr = FALSE
))
make_prerequisites <- function(target) {
  line <- make_database[startsWith(make_database, paste0(target, ":"))]
  if (length(line) == 0) {
    return(character(0))
  }
  found <- strsplit(trimws(sub("^[^:]*:", "", line[1])), "[[:space:]]+")[[1]]
  found[nzchar(found)]
}

check(
  "the scripts and Make's database were both read",
  length(scripts) >= 20 && any(grepl("^# Files", make_database))
)

known_tables <- names(table_paths())
for (path in scripts) {
  script <- script_name(path)
  inputs <- declared_inputs(path)
  unused <- setdiff(inputs$tables, inputs$symbols)
  undeclared <- setdiff(intersect(known_tables, inputs$symbols), inputs$tables)
  listed <- make_prerequisites(estimates_file(script))
  problems <- c(
    if (length(unused)) {
      paste("loads but never uses", paste(unused, collapse = ", "))
    },
    if (length(undeclared)) {
      paste("uses without loading", paste(undeclared, collapse = ", "))
    },
    if (length(setdiff(inputs$files, listed))) {
      paste(
        "reads what the Makefile does not list:",
        paste(setdiff(inputs$files, listed), collapse = ", ")
      )
    },
    if (length(setdiff(listed, inputs$files))) {
      paste(
        "is listed in the Makefile with what it does not read:",
        paste(setdiff(listed, inputs$files), collapse = ", ")
      )
    }
  )
  if (length(problems)) {
    cat(sprintf("%s %s\n", script, paste(problems, collapse = "; ")))
  }
  check(
    sprintf("%s reads what the Makefile lists for it", script),
    length(problems) == 0
  )
}
