# ---------------------------------------------------------------
# 01_load_data.R
# Load the handwriting dataset and do the basic preprocessing
# every later script (CNN, Siamese network) will reuse.
# ---------------------------------------------------------------

# Read the data: a list with $images (25000 x 28 x 28 x 1) and
# $metadata (one row per image: image_id, person_id, digit, session, replicate)
data <- readRDS("data/handwriting.rds")

# Quick sanity checks -- these should match the brief's description
dim(data$images)      # expect 25000 28 28 1
head(data$metadata)
table(data$metadata$person_id)   # should be 1250 images per writer (20 writers)
table(data$metadata$digit)       # should be 5000 images per digit (digits 0-4)
table(data$metadata$session)     # should be 5000 images per session (sessions 1-5)

# Scale pixel values from [0, 255] to [0, 1], as the brief instructs
x <- data$images / 255

# Writer ID label, 0-indexed for Keras (person_id is 1-20 in the raw data)
y_person <- data$metadata$person_id - 1L

# Keep digit and session around too -- we'll need them for exploration,
# for controlling pair construction in the Siamese network, and for the
# session-based train/test split later on.
digit   <- data$metadata$digit
session <- data$metadata$session
person  <- data$metadata$person_id
