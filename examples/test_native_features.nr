fn main():
  print("=====================================================")
  print("  TESTING NATIVE NEURON FEATURES & BUILT-INS        ")
  print("=====================================================")
  print("")

  // 1. Multi-argument print
  print("1. Multi-arg print test:", 100, 3.1415, "Success!")
  print("")

  // 2. Vector Math & Similarity
  let v1 = zeros(1, 3) + 1.0
  let v2 = zeros(1, 3) + 2.0
  let sim = cosine_sim(v1, v2)
  let dist = euclidean_dist(v1, v2)
  print("2. Math test: cosine_sim =", sim, "| euclidean_dist =", dist)
  print("")

  // 3. Native CSV Reading
  let csv_path = "C:/Users/ADMIN/quant-dashboard/backend/venv/Lib/site-packages/sklearn/datasets/data/breast_cancer.csv"
  print("3. Loading CSV dataset from:", csv_path)
  let raw_data = read_csv(csv_path)
  let r = rows(raw_data)
  let c = cols(raw_data)
  print("   Loaded shape: rows =", r, "cols =", c)
  print("")

  // 4. Slicing & Preprocessing
  let X_raw = slice_cols(raw_data, 0, 30)
  let y_col = slice_cols(raw_data, 30, 31)
  let X = min_max_scale(X_raw)
  print("4. Preprocessing: scaled", rows(X), "patients across", cols(X), "features.")
  print("")

  // 5. Native EpisodicMemory Agent Workflow
  let em_mal = EpisodicMemory(30, 200)
  let em_ben = EpisodicMemory(30, 300)

  print("5. Populating episodic memory directly in pure NEURON...")
  for i in range(455):
    let patient = get_row(X, i)
    let label = item(get_row(y_col, i))
    if label == 0.0:
      store(em_mal, patient)
    else:
      store(em_ben, patient)

  print("   Training done! Stored mal memories:", em_mal.len(), "| ben memories:", em_ben.len())
  print("")

  // 6. Native Evaluation Loop
  print("6. Evaluating 114 unseen held-out patients in pure NEURON...")
  let correct = 0
  for i in range(114):
    let idx = 455 + i
    let query = get_row(X, idx)
    let true_label = item(get_row(y_col, idx))

    let wm = em_mal.recall_weighted(query, 25)
    let wb = em_ben.recall_weighted(query, 25)

    let dm = euclidean_dist(query, wm)
    let db = euclidean_dist(query, wb)

    if dm < db:
      if true_label == 0.0:
        let correct = correct + 1
    else:
      if true_label == 1.0:
        let correct = correct + 1

  print("=====================================================")
  print("  PURE NEURON EVALUATION RESULT:                     ")
  print("  Correct:", correct, "/ 114 unseen cases")
  print("=====================================================")
