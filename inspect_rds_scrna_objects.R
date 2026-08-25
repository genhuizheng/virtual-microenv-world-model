#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  has_seurat <- requireNamespace("Seurat", quietly = TRUE)
  has_sce <- requireNamespace("SingleCellExperiment", quietly = TRUE)
  has_se <- requireNamespace("SummarizedExperiment", quietly = TRUE)
  has_matrix <- requireNamespace("Matrix", quietly = TRUE)
})

args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args) >= 1) args[[1]] else "."
output_csv <- if (length(args) >= 2) args[[2]] else file.path(input_dir, "rds_scrna_inspection_summary.csv")
clinical_csv <- sub("\\.csv$", "_clinical_columns.csv", output_csv, ignore.case = TRUE)
detailed_txt <- sub("\\.csv$", "_detailed_structure.txt", output_csv, ignore.case = TRUE)

safe_dim <- function(x) {
  out <- tryCatch(dim(x), error = function(e) NULL)
  if (is.null(out) || length(out) < 2) return(c(NA_integer_, NA_integer_))
  c(as.integer(out[[1]]), as.integer(out[[2]]))
}

safe_class <- function(x) paste(class(x), collapse = "|")

is_count_like <- function(x, n_check = 10000) {
  if (is.null(x)) return(NA)
  vals <- tryCatch({
    if (inherits(x, "sparseMatrix")) {
      x@x
    } else {
      as.numeric(x)
    }
  }, error = function(e) numeric())
  vals <- vals[is.finite(vals)]
  if (length(vals) == 0) return(NA)
  if (length(vals) > n_check) vals <- sample(vals, n_check)
  nonnegative <- mean(vals >= 0) > 0.999
  integerish <- mean(abs(vals - round(vals)) < 1e-6) > 0.999
  nonnegative && integerish
}

matrix_summary <- function(x) {
  d <- safe_dim(x)
  count_like <- is_count_like(x)
  nnz <- tryCatch({
    if (inherits(x, "sparseMatrix")) length(x@x) else sum(as.numeric(x) != 0, na.rm = TRUE)
  }, error = function(e) NA_integer_)
  list(
    n_features = d[[1]],
    n_cells = d[[2]],
    matrix_class = safe_class(x),
    nnz = nnz,
    count_like = count_like
  )
}

collapse_names <- function(x, max_n = 30) {
  if (is.null(x) || length(x) == 0) return("")
  x <- as.character(x)
  if (length(x) > max_n) {
    paste(c(x[seq_len(max_n)], paste0("...+", length(x) - max_n, " more")), collapse = ";")
  } else {
    paste(x, collapse = ";")
  }
}

clinical_patterns <- c(
  "os", "overall", "survival", "vital", "death", "dead", "deceased",
  "pfs", "progression", "progress", "recurrence", "recur", "relapse",
  "dfs", "rfs", "efs", "dss", "pfi",
  "response", "responder", "non.?responder", "recist", "orr", "bor",
  "pcr", "rcb", "pathologic", "pathological",
  "treatment", "therapy", "treated", "pretreatment", "pre.?treatment",
  "posttreatment", "post.?treatment", "baseline", "timepoint", "time_point",
  "follow.?up", "followup", "days", "months",
  "patient", "donor", "case", "sample", "subject"
)

find_clinical_columns <- function(meta) {
  if (is.null(meta) || ncol(meta) == 0) return(data.frame())
  cols <- colnames(meta)
  hit <- rep(FALSE, length(cols))
  for (pat in clinical_patterns) {
    hit <- hit | grepl(pat, cols, ignore.case = TRUE)
  }
  cols <- cols[hit]
  if (length(cols) == 0) return(data.frame())

  rows <- lapply(cols, function(col) {
    v <- meta[[col]]
    non_na <- sum(!is.na(v))
    unique_n <- length(unique(v[!is.na(v)]))
    preview <- tryCatch({
      vals <- unique(as.character(v[!is.na(v)]))
      vals <- vals[vals != ""]
      collapse_names(vals, 12)
    }, error = function(e) "")
    data.frame(
      column = col,
      class = paste(class(v), collapse = "|"),
      non_na = non_na,
      unique_n = unique_n,
      preview_values = preview,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

get_seurat_layer <- function(obj, assay, layer_name) {
  if (!has_seurat) return(NULL)
  out <- tryCatch(
    Seurat::GetAssayData(obj, assay = assay, layer = layer_name),
    error = function(e) NULL
  )
  if (!is.null(out)) return(out)

  # Fallback for older Seurat objects.
  slot_name <- layer_name
  if (layer_name %in% c("counts", "data", "scale.data")) {
    out <- tryCatch(
      Seurat::GetAssayData(obj, assay = assay, slot = slot_name),
      error = function(e) NULL
    )
  }
  out
}

get_seurat_layer_names <- function(obj, assay) {
  layers <- tryCatch(Seurat::Layers(obj[[assay]]), error = function(e) character())
  if (length(layers) == 0) {
    layers <- tryCatch(slotNames(obj[[assay]]), error = function(e) character())
    layers <- intersect(layers, c("counts", "data", "scale.data"))
  }
  as.character(layers)
}

inspect_seurat <- function(obj, path) {
  assays <- names(obj@assays)
  rows <- list()
  meta_cols <- tryCatch(colnames(obj@meta.data), error = function(e) character())

  for (assay in assays) {
    layers <- get_seurat_layer_names(obj, assay)
    counts_layer <- layers[grepl("^counts($|\\.)", layers)][1]
    data_layer <- layers[grepl("^data($|\\.)", layers)][1]
    scale_layer <- layers[grepl("^scale\\.data($|\\.)", layers)][1]

    counts <- if (!is.na(counts_layer)) get_seurat_layer(obj, assay, counts_layer) else NULL
    data <- if (!is.na(data_layer)) get_seurat_layer(obj, assay, data_layer) else NULL
    scale_data <- if (!is.na(scale_layer)) get_seurat_layer(obj, assay, scale_layer) else NULL

    counts_s <- matrix_summary(counts)
    data_s <- matrix_summary(data)
    scale_s <- matrix_summary(scale_data)

    rows[[length(rows) + 1]] <- data.frame(
      file = path,
      object_class = safe_class(obj),
      container = "Seurat",
      assay = assay,
      layers = collapse_names(layers, 80),
      selected_counts_layer = ifelse(is.na(counts_layer), "", counts_layer),
      selected_data_layer = ifelse(is.na(data_layer), "", data_layer),
      has_counts = !is.null(counts),
      counts_n_features = counts_s$n_features,
      counts_n_cells = counts_s$n_cells,
      counts_matrix_class = counts_s$matrix_class,
      counts_nnz = counts_s$nnz,
      counts_count_like = counts_s$count_like,
      has_data = !is.null(data),
      data_n_features = data_s$n_features,
      data_n_cells = data_s$n_cells,
      data_matrix_class = data_s$matrix_class,
      data_nnz = data_s$nnz,
      data_count_like = data_s$count_like,
      has_scale_data = !is.null(scale_data),
      scale_n_features = scale_s$n_features,
      scale_n_cells = scale_s$n_cells,
      reductions = collapse_names(names(obj@reductions)),
      meta_columns = collapse_names(meta_cols, 80),
      note = "",
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}

inspect_sce <- function(obj, path) {
  assay_names <- SummarizedExperiment::assayNames(obj)
  rows <- list()
  meta_cols <- tryCatch(colnames(SummarizedExperiment::colData(obj)), error = function(e) character())

  for (assay in assay_names) {
    mat <- tryCatch(SummarizedExperiment::assay(obj, assay), error = function(e) NULL)
    s <- matrix_summary(mat)
    rows[[length(rows) + 1]] <- data.frame(
      file = path,
      object_class = safe_class(obj),
      container = "SingleCellExperiment/SummarizedExperiment",
      assay = assay,
      layers = collapse_names(assay_names, 80),
      selected_counts_layer = ifelse(assay == "counts", assay, ""),
      selected_data_layer = ifelse(assay != "counts", assay, ""),
      has_counts = assay == "counts",
      counts_n_features = if (assay == "counts") s$n_features else NA_integer_,
      counts_n_cells = if (assay == "counts") s$n_cells else NA_integer_,
      counts_matrix_class = if (assay == "counts") s$matrix_class else "",
      counts_nnz = if (assay == "counts") s$nnz else NA,
      counts_count_like = if (assay == "counts") s$count_like else NA,
      has_data = assay != "counts",
      data_n_features = if (assay != "counts") s$n_features else NA_integer_,
      data_n_cells = if (assay != "counts") s$n_cells else NA_integer_,
      data_matrix_class = if (assay != "counts") s$matrix_class else "",
      data_nnz = if (assay != "counts") s$nnz else NA,
      data_count_like = if (assay != "counts") s$count_like else NA,
      has_scale_data = FALSE,
      scale_n_features = NA_integer_,
      scale_n_cells = NA_integer_,
      reductions = "",
      meta_columns = collapse_names(meta_cols, 80),
      note = "",
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}

inspect_matrix_or_df <- function(obj, path) {
  s <- matrix_summary(obj)
  data.frame(
    file = path,
    object_class = safe_class(obj),
    container = "matrix/data.frame/other",
    assay = "object",
    layers = "",
    selected_counts_layer = ifelse(isTRUE(s$count_like), "object", ""),
    selected_data_layer = "object",
    has_counts = s$count_like,
    counts_n_features = s$n_features,
    counts_n_cells = s$n_cells,
    counts_matrix_class = s$matrix_class,
    counts_nnz = s$nnz,
    counts_count_like = s$count_like,
    has_data = TRUE,
    data_n_features = s$n_features,
    data_n_cells = s$n_cells,
    data_matrix_class = s$matrix_class,
    data_nnz = s$nnz,
    data_count_like = s$count_like,
    has_scale_data = FALSE,
    scale_n_features = NA_integer_,
    scale_n_cells = NA_integer_,
    reductions = "",
    meta_columns = "",
    note = "Plain object. Need documentation to know whether values are raw counts or processed expression.",
    stringsAsFactors = FALSE
  )
}

inspect_one <- function(path) {
  message("Inspecting: ", path)
  obj <- tryCatch(readRDS(path), error = function(e) e)
  if (inherits(obj, "error")) {
    return(data.frame(
      file = path,
      object_class = "READ_ERROR",
      container = "",
      assay = "",
      layers = "",
      selected_counts_layer = "",
      selected_data_layer = "",
      has_counts = NA,
      counts_n_features = NA_integer_,
      counts_n_cells = NA_integer_,
      counts_matrix_class = "",
      counts_nnz = NA,
      counts_count_like = NA,
      has_data = NA,
      data_n_features = NA_integer_,
      data_n_cells = NA_integer_,
      data_matrix_class = "",
      data_nnz = NA,
      data_count_like = NA,
      has_scale_data = NA,
      scale_n_features = NA_integer_,
      scale_n_cells = NA_integer_,
      reductions = "",
      meta_columns = "",
      note = obj$message,
      stringsAsFactors = FALSE
    ))
  }

  if (has_seurat && inherits(obj, "Seurat")) return(inspect_seurat(obj, path))
  if (has_sce && inherits(obj, "SingleCellExperiment")) return(inspect_sce(obj, path))
  if (has_se && inherits(obj, "SummarizedExperiment")) return(inspect_sce(obj, path))
  inspect_matrix_or_df(obj, path)
}

cat_line <- function(..., file) {
  cat(..., "\n", file = file, append = TRUE, sep = "")
}

capture_to_text <- function(expr) {
  paste(capture.output(expr), collapse = "\n")
}

preview_vector <- function(x, n = 20) {
  x <- tryCatch(as.character(x), error = function(e) character())
  x <- x[!is.na(x)]
  x <- unique(x)
  if (length(x) == 0) return("")
  paste(head(x, n), collapse = "; ")
}

write_metadata_profile <- function(meta, file) {
  if (is.null(meta) || ncol(meta) == 0) {
    cat_line("Metadata: none detected", file = file)
    return(invisible(NULL))
  }
  cat_line("Metadata dimensions: ", nrow(meta), " cells x ", ncol(meta), " columns", file = file)
  cat_line("Metadata columns:", file = file)
  for (col in colnames(meta)) {
    v <- meta[[col]]
    non_na <- sum(!is.na(v))
    unique_n <- length(unique(v[!is.na(v)]))
    class_v <- paste(class(v), collapse = "|")
    preview <- preview_vector(v, 12)
    cat_line("  - ", col, " | class=", class_v, " | non_na=", non_na, " | unique=", unique_n, " | preview=", preview, file = file)
  }
}

write_matrix_profile <- function(mat, label, file) {
  if (is.null(mat)) {
    cat_line("  ", label, ": missing", file = file)
    return(invisible(NULL))
  }
  d <- safe_dim(mat)
  count_like <- is_count_like(mat)
  cat_line("  ", label, ": ", d[[1]], " features x ", d[[2]], " cells | class=", safe_class(mat), " | count_like=", count_like, file = file)
  vals <- tryCatch({
    if (inherits(mat, "sparseMatrix")) mat@x else as.numeric(mat)
  }, error = function(e) numeric())
  vals <- vals[is.finite(vals)]
  if (length(vals) > 0) {
    if (length(vals) > 100000) vals <- sample(vals, 100000)
    qs <- quantile(vals, probs = c(0, 0.25, 0.5, 0.75, 0.99, 1), na.rm = TRUE)
    cat_line("    value_quantiles(sampled): ", paste(names(qs), signif(qs, 5), sep = "=", collapse = "; "), file = file)
  }
}

write_detailed_report_one <- function(path, file) {
  cat_line(strrep("=", 100), file = file)
  cat_line("FILE: ", path, file = file)
  cat_line(strrep("=", 100), file = file)

  obj <- tryCatch(readRDS(path), error = function(e) e)
  if (inherits(obj, "error")) {
    cat_line("READ ERROR: ", obj$message, file = file)
    return(invisible(NULL))
  }

  cat_line("Object class: ", safe_class(obj), file = file)
  cat_line("Object size: ", format(object.size(obj), units = "auto"), file = file)
  cat_line("Top-level names/slots:", file = file)
  cat_line(capture_to_text(str(obj, max.level = 1)), file = file)
  cat_line("", file = file)

  if (has_seurat && inherits(obj, "Seurat")) {
    cat_line("SEURAT SUMMARY", file = file)
    cat_line("Default assay: ", tryCatch(Seurat::DefaultAssay(obj), error = function(e) ""), file = file)
    cat_line("Assays: ", paste(names(obj@assays), collapse = "; "), file = file)
    cat_line("Reductions: ", paste(names(obj@reductions), collapse = "; "), file = file)
    cat_line("Graphs: ", paste(names(obj@graphs), collapse = "; "), file = file)
    cat_line("Images: ", paste(names(obj@images), collapse = "; "), file = file)
    cat_line("Commands: ", paste(names(obj@commands), collapse = "; "), file = file)
    cat_line("Project: ", tryCatch(obj@project.name, error = function(e) ""), file = file)
    cat_line("", file = file)

    for (assay in names(obj@assays)) {
      cat_line("ASSAY: ", assay, file = file)
      cat_line("  assay class: ", safe_class(obj[[assay]]), file = file)
      layers <- get_seurat_layer_names(obj, assay)
      cat_line("  layers/slots: ", paste(layers, collapse = "; "), file = file)
      for (layer in layers) {
        mat <- get_seurat_layer(obj, assay, layer)
        write_matrix_profile(mat, paste0("layer ", layer), file)
      }
      features <- tryCatch(rownames(obj[[assay]]), error = function(e) character())
      cat_line("  feature preview: ", paste(head(features, 20), collapse = "; "), file = file)
      cat_line("", file = file)
    }

    cat_line("REDUCTIONS DETAIL", file = file)
    for (red in names(obj@reductions)) {
      emb <- tryCatch(Seurat::Embeddings(obj, reduction = red), error = function(e) NULL)
      d <- safe_dim(emb)
      cat_line("  - ", red, ": ", d[[1]], " cells x ", d[[2]], " dims", file = file)
    }
    cat_line("", file = file)

    cat_line("METADATA DETAIL", file = file)
    write_metadata_profile(tryCatch(obj@meta.data, error = function(e) NULL), file)
    cat_line("", file = file)

    clinical_hits <- find_clinical_columns(tryCatch(obj@meta.data, error = function(e) NULL))
    cat_line("CLINICAL-LIKE METADATA HITS", file = file)
    if (nrow(clinical_hits) == 0) {
      cat_line("  none detected", file = file)
    } else {
      cat_line(capture_to_text(print(clinical_hits, row.names = FALSE)), file = file)
    }
    cat_line("", file = file)
    return(invisible(NULL))
  }

  if ((has_sce && inherits(obj, "SingleCellExperiment")) || (has_se && inherits(obj, "SummarizedExperiment"))) {
    cat_line("SCE/SUMMARIZEDEXPERIMENT SUMMARY", file = file)
    cat_line("Assays: ", paste(SummarizedExperiment::assayNames(obj), collapse = "; "), file = file)
    for (assay in SummarizedExperiment::assayNames(obj)) {
      mat <- tryCatch(SummarizedExperiment::assay(obj, assay), error = function(e) NULL)
      write_matrix_profile(mat, paste0("assay ", assay), file)
    }
    cat_line("METADATA DETAIL", file = file)
    write_metadata_profile(tryCatch(as.data.frame(SummarizedExperiment::colData(obj)), error = function(e) NULL), file)
    cat_line("", file = file)
    return(invisible(NULL))
  }

  cat_line("PLAIN OBJECT SUMMARY", file = file)
  if (is.matrix(obj) || is.data.frame(obj) || inherits(obj, "sparseMatrix")) {
    write_matrix_profile(obj, "object", file)
  }
  cat_line("", file = file)
}

paths <- list.files(input_dir, pattern = "\\.rds$", full.names = TRUE, recursive = TRUE, ignore.case = TRUE)
if (length(paths) == 0) {
  stop("No .rds files found in: ", input_dir)
}

summary <- do.call(rbind, lapply(paths, inspect_one))
write.csv(summary, output_csv, row.names = FALSE)

if (file.exists(detailed_txt)) file.remove(detailed_txt)
cat_line("RDS scRNA detailed structure report", file = detailed_txt)
cat_line("Generated: ", as.character(Sys.time()), file = detailed_txt)
cat_line("Input dir: ", input_dir, file = detailed_txt)
cat_line("Files found: ", length(paths), file = detailed_txt)
cat_line("", file = detailed_txt)
invisible(lapply(paths, write_detailed_report_one, file = detailed_txt))

clinical_rows <- list()
for (path in paths) {
  obj <- tryCatch(readRDS(path), error = function(e) NULL)
  if (is.null(obj)) next
  meta <- NULL
  if (has_seurat && inherits(obj, "Seurat")) {
    meta <- tryCatch(obj@meta.data, error = function(e) NULL)
  } else if ((has_sce && inherits(obj, "SingleCellExperiment")) || (has_se && inherits(obj, "SummarizedExperiment"))) {
    meta <- tryCatch(as.data.frame(SummarizedExperiment::colData(obj)), error = function(e) NULL)
  }
  hits <- find_clinical_columns(meta)
  if (nrow(hits) > 0) {
    hits$file <- path
    hits <- hits[, c("file", "column", "class", "non_na", "unique_n", "preview_values")]
    clinical_rows[[length(clinical_rows) + 1]] <- hits
  }
}
if (length(clinical_rows) > 0) {
  clinical_summary <- do.call(rbind, clinical_rows)
  write.csv(clinical_summary, clinical_csv, row.names = FALSE)
} else {
  clinical_summary <- data.frame()
  write.csv(data.frame(note = "No clinical-like metadata columns found."), clinical_csv, row.names = FALSE)
}

cat("\nDone.\n")
cat("Inspected files:", length(paths), "\n")
cat("Output CSV:", output_csv, "\n\n")
cat("Clinical-column CSV:", clinical_csv, "\n\n")
cat("Detailed TXT:", detailed_txt, "\n\n")

print(summary[, c(
  "file",
  "object_class",
  "container",
  "assay",
  "layers",
  "selected_counts_layer",
  "selected_data_layer",
  "has_counts",
  "counts_n_features",
  "counts_n_cells",
  "counts_count_like",
  "has_data",
  "data_count_like"
)], row.names = FALSE)

if (nrow(clinical_summary) > 0) {
  cat("\nPotential clinical/sample metadata columns:\n")
  print(clinical_summary, row.names = FALSE)
}
