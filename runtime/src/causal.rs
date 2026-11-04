/// Structural Causal Model (SCM) Engine & Causal Discovery (PC Algorithm).
///
/// Implements Bayesian conditioning (observe), do-calculus interventions (intervene),
/// 3-step counterfactual logic (Abduction-Action-Prediction), and skeleton/orientation PC algorithm.

use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Clone, Debug, PartialEq)]
pub struct IdentificationResult {
    pub identifiable: bool,
    pub method: String,
    pub formula: String,
    pub adjustment_set: Vec<String>,
    pub hedge: Option<Vec<String>>,
    /// Outer and inner districts witnessing non-identifiability, when ID returns a hedge.
    pub hedge_outer: Option<Vec<String>>,
    pub hedge_inner: Option<Vec<String>>,
    /// Structured symbolic formula, when the effect is identifiable.
    pub formula_ast: Option<IdentificationFormula>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum IdentificationExpr {
    Kernel {
        name: String,
        variables: Vec<String>,
        given: Vec<String>,
    },
    Sum {
        variables: Vec<String>,
        expression: Box<IdentificationExpr>,
    },
    Product(Vec<IdentificationExpr>),
    Quotient {
        numerator: Box<IdentificationExpr>,
        denominator: Box<IdentificationExpr>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct IdentificationDefinition {
    pub name: String,
    pub variables: Vec<String>,
    pub given: Vec<String>,
    pub expression: IdentificationExpr,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IdentificationFormula {
    pub outcomes: Vec<String>,
    pub treatments: Vec<String>,
    pub conditions: Vec<String>,
    pub definitions: Vec<IdentificationDefinition>,
    pub expression: IdentificationExpr,
}

/// Finite categorical observations used to evaluate an identified formula.
#[derive(Clone, Debug, PartialEq)]
pub struct CausalDataset {
    variables: Vec<String>,
    rows: Vec<Vec<String>>,
    name_to_index: HashMap<String, usize>,
    categories: HashMap<String, Vec<String>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OutcomeProbability {
    pub values: Vec<String>,
    pub probability: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CausalEstimate {
    pub method: String,
    pub formula: String,
    pub formula_ast: IdentificationFormula,
    pub sample_size: usize,
    pub distribution: Vec<OutcomeProbability>,
}

impl CausalDataset {
    pub fn new(variables: Vec<String>, rows: Vec<Vec<String>>) -> Result<Self, String> {
        if variables.is_empty() {
            return Err("CausalDataset requires at least one variable".into());
        }
        if variables.iter().any(|name| name.trim().is_empty()) {
            return Err("CausalDataset variable names cannot be empty".into());
        }
        if variables.iter().collect::<HashSet<_>>().len() != variables.len() {
            return Err("CausalDataset variable names must be unique".into());
        }
        if rows.is_empty() {
            return Err("CausalDataset requires at least one observation row".into());
        }
        for (row_index, row) in rows.iter().enumerate() {
            if row.len() != variables.len() {
                return Err(format!(
                    "CausalDataset row {} has {} values; expected {}",
                    row_index,
                    row.len(),
                    variables.len()
                ));
            }
            if row.iter().any(|value| value.is_empty()) {
                return Err(format!("CausalDataset row {} contains an empty category", row_index));
            }
        }

        let name_to_index: HashMap<String, usize> = variables.iter().cloned()
            .enumerate().map(|(index, name)| (name, index)).collect();
        let mut categories: HashMap<String, Vec<String>> = variables.iter()
            .map(|name| (name.clone(), Vec::new())).collect();
        for row in &rows {
            for (index, value) in row.iter().enumerate() {
                let values = categories.get_mut(&variables[index]).expect("category vector exists");
                if !values.contains(value) {
                    values.push(value.clone());
                }
            }
        }
        Ok(Self { variables, rows, name_to_index, categories })
    }

    pub fn variables(&self) -> &[String] {
        &self.variables
    }

    pub fn rows(&self) -> &[Vec<String>] {
        &self.rows
    }

    pub fn categories(&self, variable: &str) -> Result<&[String], String> {
        self.categories.get(variable).map(Vec::as_slice)
            .ok_or_else(|| format!("Dataset does not contain variable '{}'", variable))
    }

    fn contains_category(&self, variable: &str, category: &str) -> Result<bool, String> {
        Ok(self.categories(variable)?.contains(&category.to_string()))
    }

    fn empirical_conditional(
        &self,
        variables: &[String],
        given: &[String],
        assignment: &HashMap<String, String>,
    ) -> Result<f64, String> {
        let targets = variables.iter().map(|name| {
            let data_name = name.strip_suffix('\'').unwrap_or(name);
            let index = self.name_to_index.get(data_name)
                .ok_or_else(|| format!("Dataset does not contain formula variable '{}'", name))?;
            let value = assignment.get(name)
                .ok_or_else(|| format!("No category was assigned to formula variable '{}'", name))?;
            Ok((*index, value.as_str()))
        }).collect::<Result<Vec<_>, String>>()?;
        let conditions = given.iter().map(|name| {
            let data_name = name.strip_suffix('\'').unwrap_or(name);
            let index = self.name_to_index.get(data_name)
                .ok_or_else(|| format!("Dataset does not contain formula variable '{}'", name))?;
            let value = assignment.get(name)
                .ok_or_else(|| format!("No category was assigned to formula variable '{}'", name))?;
            Ok((*index, value.as_str()))
        }).collect::<Result<Vec<_>, String>>()?;

        let mut denominator = 0usize;
        let mut numerator = 0usize;
        'row: for row in &self.rows {
            for &(index, value) in &conditions {
                if row[index] != value { continue 'row; }
            }
            denominator += 1;
            for &(index, value) in &targets {
                if row[index] != value { continue 'row; }
            }
            numerator += 1;
        }
        if denominator == 0 {
            return Ok(0.0);
        }
        Ok(numerator as f64 / denominator as f64)
    }
}

impl IdentificationExpr {
    pub fn render(&self) -> String {
        match self {
            Self::Kernel { name, variables, given } => {
                if given.is_empty() {
                    format!("{}({})", name, variables.join(", "))
                } else {
                    format!("{}({} | {})", name, variables.join(", "), given.join(", "))
                }
            }
            Self::Sum { variables, expression } => {
                format!("sum_{{{}}} {}", variables.join(", "), expression.render())
            }
            Self::Product(factors) => factors.iter()
                .map(|factor| format!("({})", factor.render()))
                .collect::<Vec<_>>()
                .join(" * "),
            Self::Quotient { numerator, denominator } => {
                format!("({}) / ({})", numerator.render(), denominator.render())
            }
        }
    }
}

impl IdentificationFormula {
    pub fn render(&self) -> String {
        let mut lines = self.definitions.iter().map(|definition| {
            let lhs = if definition.given.is_empty() {
                format!("{}({})", definition.name, definition.variables.join(", "))
            } else {
                format!("{}({} | {})", definition.name, definition.variables.join(", "), definition.given.join(", "))
            };
            format!("{} := {}", lhs, definition.expression.render())
        }).collect::<Vec<_>>();
        let mut conditions = vec![format!("do({})", self.treatments.join(", "))];
        if !self.conditions.is_empty() {
            conditions.push(self.conditions.join(", "));
        }
        lines.push(format!("P({} | {}) = {}", self.outcomes.join(", "), conditions.join(", "), self.expression.render()));
        lines.join("\n")
    }

    /// Evaluate this formula over a finite categorical dataset and return its
    /// distribution over the requested outcomes.
    pub fn evaluate_distribution(
        &self,
        dataset: &CausalDataset,
        interventions: &HashMap<String, String>,
        conditions: &HashMap<String, String>,
    ) -> Result<Vec<OutcomeProbability>, String> {
        if self.outcomes.is_empty() {
            return Err("Cannot evaluate a formula with no outcomes".into());
        }
        let expected_treatments: HashSet<&str> = self.treatments.iter().map(String::as_str).collect();
        let actual_treatments: HashSet<&str> = interventions.keys().map(String::as_str).collect();
        if expected_treatments != actual_treatments {
            return Err(format!(
                "Intervention values must be supplied for exactly [{}]",
                self.treatments.join(", ")
            ));
        }
        let expected_conditions: HashSet<&str> = self.conditions.iter().map(String::as_str).collect();
        let actual_conditions: HashSet<&str> = conditions.keys().map(String::as_str).collect();
        if expected_conditions != actual_conditions {
            return Err(format!(
                "Condition values must be supplied for exactly [{}]",
                self.conditions.join(", ")
            ));
        }

        let mut base_assignment = HashMap::new();
        for (variable, value) in interventions.iter().chain(conditions.iter()) {
            if !dataset.contains_category(variable, value)? {
                return Err(format!(
                    "Dataset has no observations for category '{}' of variable '{}'",
                    value, variable
                ));
            }
            base_assignment.insert(variable.clone(), value.clone());
        }

        let mut assignments = vec![base_assignment];
        for outcome in &self.outcomes {
            let categories = dataset.categories(outcome)?;
            let new_len = assignments.len().checked_mul(categories.len())
                .ok_or_else(|| "Outcome distribution has too many category combinations".to_string())?;
            if new_len > 100_000 {
                return Err("Outcome distribution exceeds the 100000-combination limit".into());
            }
            let mut expanded = Vec::with_capacity(new_len);
            for assignment in &assignments {
                for category in categories {
                    let mut next = assignment.clone();
                    next.insert(outcome.clone(), category.clone());
                    expanded.push(next);
                }
            }
            assignments = expanded;
        }

        let mut evaluator = FormulaEvaluator::new(dataset, &self.definitions);
        let mut distribution = Vec::with_capacity(assignments.len());
        let mut total = 0.0;
        for assignment in assignments {
            let probability = evaluator.evaluate(&self.expression, &assignment)?;
            if !probability.is_finite() || probability < -1e-10 {
                return Err(format!("Formula produced an invalid probability: {}", probability));
            }
            let probability = probability.max(0.0);
            total += probability;
            distribution.push(OutcomeProbability {
                values: self.outcomes.iter().map(|variable| {
                    assignment.get(variable).cloned().expect("outcome value was assigned")
                }).collect(),
                probability,
            });
        }
        if !total.is_finite() || total <= 1e-12 {
            return Err("No empirical support for these intervention and condition values (positivity failure)".into());
        }
        for entry in &mut distribution {
            entry.probability /= total;
        }
        Ok(distribution)
    }
}

struct FormulaEvaluator<'a> {
    dataset: &'a CausalDataset,
    definitions: HashMap<&'a str, &'a IdentificationDefinition>,
    work: usize,
}

impl<'a> FormulaEvaluator<'a> {
    fn new(dataset: &'a CausalDataset, definitions: &'a [IdentificationDefinition]) -> Self {
        Self {
            dataset,
            definitions: definitions.iter().map(|definition| (definition.name.as_str(), definition)).collect(),
            work: 0,
        }
    }

    fn evaluate(
        &mut self,
        expression: &IdentificationExpr,
        assignment: &HashMap<String, String>,
    ) -> Result<f64, String> {
        self.work += 1;
        if self.work > 10_000_000 {
            return Err("Formula evaluation exceeded the 10000000-operation limit".into());
        }
        match expression {
            IdentificationExpr::Kernel { name, variables, given } if name == "P" => {
                self.dataset.empirical_conditional(variables, given, assignment)
            }
            IdentificationExpr::Kernel { name, variables, given } => {
                self.evaluate_defined_kernel(name, variables, given, assignment)
            }
            IdentificationExpr::Sum { variables, expression } => {
                self.sum_over(variables, expression, 0, assignment)
            }
            IdentificationExpr::Product(factors) => {
                let mut product = 1.0;
                for factor in factors {
                    product *= self.evaluate(factor, assignment)?;
                }
                Ok(product)
            }
            IdentificationExpr::Quotient { numerator, denominator } => {
                let numerator = self.evaluate(numerator, assignment)?;
                let denominator = self.evaluate(denominator, assignment)?;
                if !denominator.is_finite() || denominator <= 1e-12 {
                    return Err("Formula quotient has zero empirical denominator (positivity failure)".into());
                }
                Ok(numerator / denominator)
            }
        }
    }

    fn sum_over(
        &mut self,
        variables: &[String],
        expression: &IdentificationExpr,
        index: usize,
        assignment: &HashMap<String, String>,
    ) -> Result<f64, String> {
        if index == variables.len() {
            return self.evaluate(expression, assignment);
        }
        let variable = &variables[index];
        let data_name = variable.strip_suffix('\'').unwrap_or(variable);
        let categories = self.dataset.categories(data_name)?.to_vec();
        let mut total = 0.0;
        for category in categories {
            let mut next = assignment.clone();
            next.insert(variable.clone(), category);
            total += self.sum_over(variables, expression, index + 1, &next)?;
        }
        Ok(total)
    }

    fn evaluate_defined_kernel(
        &mut self,
        name: &str,
        variables: &[String],
        given: &[String],
        assignment: &HashMap<String, String>,
    ) -> Result<f64, String> {
        let definition = self.definitions.get(name).copied()
            .ok_or_else(|| format!("Formula references undefined probability kernel '{}'", name))?;
        let scope: HashSet<&str> = definition.variables.iter().map(String::as_str).collect();
        let context: HashSet<&str> = definition.given.iter().map(String::as_str).collect();
        if variables.iter().any(|variable| !scope.contains(variable.as_str())) {
            return Err(format!("Kernel '{}' was queried for variables outside its scope", name));
        }
        if given.iter().any(|variable| {
            !scope.contains(variable.as_str()) && !context.contains(variable.as_str())
        }) {
            return Err(format!("Kernel '{}' was conditioned on a variable outside its scope", name));
        }
        let target_set: HashSet<&str> = variables.iter().map(String::as_str).collect();
        if given.iter().any(|variable| target_set.contains(variable.as_str())) {
            return Err(format!("Kernel '{}' has overlapping target and condition variables", name));
        }
        for variable in &definition.given {
            if !assignment.contains_key(variable) {
                return Err(format!("Kernel '{}' requires context value for '{}'", name, variable));
            }
        }

        let given_set: HashSet<&str> = given.iter().map(String::as_str).collect();
        let numerator_fixed: HashSet<&str> = target_set.union(&given_set).copied().collect();
        let numerator_sum: Vec<String> = definition.variables.iter()
            .filter(|variable| !numerator_fixed.contains(variable.as_str())).cloned().collect();
        let denominator_sum: Vec<String> = definition.variables.iter()
            .filter(|variable| !given_set.contains(variable.as_str())).cloned().collect();
        let numerator = self.sum_over(&numerator_sum, &definition.expression, 0, assignment)?;
        let denominator = self.sum_over(&denominator_sum, &definition.expression, 0, assignment)?;
        if denominator <= 1e-12 {
            return Ok(0.0);
        }
        Ok(numerator / denominator)
    }
}

#[derive(Clone, Debug)]
pub struct CausalModel {
    pub names: Vec<String>,
    pub name_to_idx: HashMap<String, usize>,
    /// Adjacency weight matrix (parents -> children)
    /// weights[i][j] is the causal weight of i on j.
    pub weights: Vec<Vec<f64>>,
    /// Exogenous noise variances
    pub noise_variances: Vec<f64>,
    /// Exogenous noise means
    pub noise_means: Vec<f64>,
    /// Off-diagonal exogenous noise covariances, keyed by sorted variable indices.
    /// A bidirected edge without an explicit covariance remains topological only.
    pub noise_covariances: HashMap<(usize, usize), f64>,
    /// Bi-directed / confounded edges (u, v) with u < v
    pub bidirected: Vec<(usize, usize)>,
}

#[derive(Clone, Debug)]
struct IdGraph {
    nodes: Vec<usize>,
    directed: Vec<(usize, usize)>,
    bidirected: Vec<(usize, usize)>,
}

impl IdGraph {
    fn from_model(model: &CausalModel) -> Result<Self, String> {
        let n = model.names.len();
        if model.weights.len() != n || model.weights.iter().any(|row| row.len() != n) {
            return Err("Causal graph adjacency matrix must be square and match its variables".into());
        }
        if model.names.iter().collect::<HashSet<_>>().len() != n {
            return Err("Causal graph variable names must be unique".into());
        }

        let mut directed = Vec::new();
        for from in 0..n {
            for to in 0..n {
                if model.weights[from][to].abs() > 1e-12 {
                    directed.push((from, to));
                }
            }
        }
        let mut bidirected = model.bidirected.clone();
        if bidirected.iter().any(|&(a, b)| a >= n || b >= n || a == b) {
            return Err("Causal graph contains an invalid bidirected edge".into());
        }
        for edge in &mut bidirected {
            if edge.0 > edge.1 {
                *edge = (edge.1, edge.0);
            }
        }
        bidirected.sort_unstable();
        bidirected.dedup();

        Ok(Self {
            nodes: (0..n).collect(),
            directed,
            bidirected,
        })
    }

    fn induced(&self, keep: &HashSet<usize>) -> Self {
        Self {
            nodes: self.nodes.iter().copied().filter(|node| keep.contains(node)).collect(),
            directed: self.directed.iter().copied()
                .filter(|(from, to)| keep.contains(from) && keep.contains(to)).collect(),
            bidirected: self.bidirected.iter().copied()
                .filter(|(a, b)| keep.contains(a) && keep.contains(b)).collect(),
        }
    }

    fn mutilated(&self, cut_incoming: &HashSet<usize>, cut_outgoing: &HashSet<usize>) -> Self {
        Self {
            nodes: self.nodes.clone(),
            directed: self.directed.iter().copied()
                .filter(|(from, to)| !cut_outgoing.contains(from) && !cut_incoming.contains(to))
                .collect(),
            bidirected: self.bidirected.clone(),
        }
    }

    fn m_separated(&self, x: &[usize], y: &[usize], z: &[usize]) -> bool {
        let z_set: HashSet<usize> = z.iter().copied().collect();
        let x_set: HashSet<usize> = x.iter().copied().collect();
        let y_set: HashSet<usize> = y.iter().copied().collect();
        // A collider is active when it is an ancestor of a conditioned node.
        let ancestors_of_z = self.ancestors(z, &HashSet::new());

        for &start in &x_set {
            if !self.nodes.contains(&start) {
                continue;
            }
            let mut visited = HashSet::from([start]);
            let mut path = vec![start];
            let mut edges = Vec::new();
            if self.find_active_trail(
                start, &y_set, &z_set, &ancestors_of_z, &mut visited, &mut path, &mut edges,
            ) {
                return false;
            }
        }
        true
    }

    fn find_active_trail(
        &self,
        current: usize,
        targets: &HashSet<usize>,
        conditioned: &HashSet<usize>,
        ancestors_of_conditioned: &HashSet<usize>,
        visited: &mut HashSet<usize>,
        path: &mut Vec<usize>,
        edges: &mut Vec<(bool, bool)>,
    ) -> bool {
        if path.len() > 1 && targets.contains(&current)
            && self.trail_is_active(path, edges, conditioned, ancestors_of_conditioned)
        {
            return true;
        }

        let mut neighbors = Vec::new();
        for &(from, to) in &self.directed {
            if from == current && !visited.contains(&to) {
                neighbors.push((to, false, true));
            }
            if to == current && !visited.contains(&from) {
                neighbors.push((from, true, false));
            }
        }
        for &(a, b) in &self.bidirected {
            if a == current && !visited.contains(&b) {
                neighbors.push((b, true, true));
            } else if b == current && !visited.contains(&a) {
                neighbors.push((a, true, true));
            }
        }

        for (next, arrow_into_current, arrow_into_next) in neighbors {
            visited.insert(next);
            path.push(next);
            edges.push((arrow_into_current, arrow_into_next));
            let active_prefix = path.len() < 3 || self.internal_node_is_active(
                path.len() - 2, path, edges, conditioned, ancestors_of_conditioned,
            );
            if active_prefix && self.find_active_trail(
                next, targets, conditioned, ancestors_of_conditioned, visited, path, edges,
            ) {
                return true;
            }
            edges.pop();
            path.pop();
            visited.remove(&next);
        }
        false
    }

    fn internal_node_is_active(
        &self,
        index: usize,
        path: &[usize],
        edges: &[(bool, bool)],
        conditioned: &HashSet<usize>,
        ancestors_of_conditioned: &HashSet<usize>,
    ) -> bool {
        let node = path[index];
        let collider = edges[index - 1].1 && edges[index].0;
        if collider {
            ancestors_of_conditioned.contains(&node)
        } else {
            !conditioned.contains(&node)
        }
    }

    fn trail_is_active(
        &self,
        path: &[usize],
        edges: &[(bool, bool)],
        conditioned: &HashSet<usize>,
        ancestors_of_conditioned: &HashSet<usize>,
    ) -> bool {
        (1..path.len() - 1).all(|index| self.internal_node_is_active(
            index, path, edges, conditioned, ancestors_of_conditioned,
        ))
    }

    fn c_components(&self) -> Vec<Vec<usize>> {
        let mut visited = HashSet::new();
        let mut components = Vec::new();
        for &start in &self.nodes {
            if !visited.insert(start) {
                continue;
            }
            let mut component = vec![start];
            let mut queue = VecDeque::new();
            queue.push_back(start);
            while let Some(current) = queue.pop_front() {
                for &(a, b) in &self.bidirected {
                    let neighbor = if a == current { Some(b) } else if b == current { Some(a) } else { None };
                    if let Some(neighbor) = neighbor {
                        if self.nodes.binary_search(&neighbor).is_ok() && visited.insert(neighbor) {
                            component.push(neighbor);
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
            component.sort_unstable();
            components.push(component);
        }
        components.sort_by_key(|component| component.first().copied().unwrap_or(usize::MAX));
        components
    }

    fn ancestors(&self, targets: &[usize], cut_incoming: &HashSet<usize>) -> HashSet<usize> {
        let mut ancestors = HashSet::new();
        let mut queue: VecDeque<usize> = targets.iter().copied().collect();
        while let Some(node) = queue.pop_front() {
            if !ancestors.insert(node) {
                continue;
            }
            if cut_incoming.contains(&node) {
                continue;
            }
            for &(parent, child) in &self.directed {
                if child == node {
                    queue.push_back(parent);
                }
            }
        }
        ancestors
    }

    fn topological_order(&self) -> Option<Vec<usize>> {
        let mut indegree: HashMap<usize, usize> = self.nodes.iter().map(|&node| (node, 0)).collect();
        for &(from, to) in &self.directed {
            *indegree.get_mut(&to)? += 1;
            if !indegree.contains_key(&from) {
                return None;
            }
        }
        let mut ready: Vec<usize> = indegree.iter().filter_map(|(&node, &degree)| (degree == 0).then_some(node)).collect();
        ready.sort_unstable();
        let mut order = Vec::with_capacity(self.nodes.len());
        while !ready.is_empty() {
            let node = ready.remove(0);
            order.push(node);
            for &(from, to) in &self.directed {
                if from == node {
                    let degree = indegree.get_mut(&to)?;
                    *degree -= 1;
                    if *degree == 0 {
                        let at = ready.binary_search(&to).unwrap_or_else(|at| at);
                        ready.insert(at, to);
                    }
                }
            }
        }
        (order.len() == self.nodes.len()).then_some(order)
    }
}

#[derive(Clone, Debug)]
struct IdKernel {
    name: String,
    scope: Vec<usize>,
    context: Vec<usize>,
}

#[derive(Debug)]
enum IdFailure {
    Invalid(String),
    Hedge { outer: Vec<usize>, inner: Vec<usize> },
}

struct IdContext<'a> {
    names: &'a [String],
    definitions: Vec<IdentificationDefinition>,
    next_kernel: usize,
}

impl<'a> IdContext<'a> {
    fn new(names: &'a [String]) -> Self {
        Self { names, definitions: Vec::new(), next_kernel: 0 }
    }

    fn node_name_list(&self, nodes: &[usize]) -> Vec<String> {
        nodes.iter().map(|&node| self.names[node].clone()).collect()
    }

    fn kernel_expr(&self, name: &str, variables: &[usize], given: &[usize]) -> IdentificationExpr {
        IdentificationExpr::Kernel {
            name: name.into(),
            variables: self.node_name_list(variables),
            given: self.node_name_list(given),
        }
    }

    fn conditional_expr(&self, kernel: &IdKernel, node: usize, preceding: &[usize]) -> IdentificationExpr {
        let mut conditions = preceding.to_vec();
        conditions.extend(kernel.context.iter().copied());
        conditions.sort_unstable();
        conditions.dedup();
        self.kernel_expr(&kernel.name, &[node], &conditions)
    }

    fn fresh_kernel(&mut self, prefix: &str, scope: &[usize], context: &[usize]) -> IdKernel {
        let name = format!("{}{}", prefix, self.next_kernel);
        self.next_kernel += 1;
        IdKernel { name, scope: scope.to_vec(), context: context.to_vec() }
    }

    fn marginal(&mut self, source: &IdKernel, keep: &[usize]) -> IdKernel {
        let kernel = self.fresh_kernel("M", keep, &source.context);
        let keep_set: HashSet<usize> = keep.iter().copied().collect();
        let eliminated: Vec<usize> = source.scope.iter().copied().filter(|node| !keep_set.contains(node)).collect();
        let source_expr = self.kernel_expr(&source.name, &source.scope, &source.context);
        let expression = if eliminated.is_empty() {
            source_expr
        } else {
            IdentificationExpr::Sum {
                variables: self.node_name_list(&eliminated),
                expression: Box::new(source_expr),
            }
        };
        self.definitions.push(IdentificationDefinition {
            name: kernel.name.clone(),
            variables: self.node_name_list(keep),
            given: self.node_name_list(&kernel.context),
            expression,
        });
        kernel
    }

    fn q_factor(&mut self, source: &IdKernel, selected: &[usize], order: &[usize]) -> IdKernel {
        let selected_set: HashSet<usize> = selected.iter().copied().collect();
        let mut factors = Vec::new();
        let mut preceding = Vec::new();
        let mut context = source.context.clone();
        for &node in order {
            if selected_set.contains(&node) {
                factors.push(self.conditional_expr(source, node, &preceding));
                context.extend(preceding.iter().copied().filter(|prior| !selected_set.contains(prior)));
            }
            preceding.push(node);
        }
        context.sort_unstable();
        context.dedup();
        let kernel = self.fresh_kernel("Q", selected, &context);
        self.definitions.push(IdentificationDefinition {
            name: kernel.name.clone(),
            variables: self.node_name_list(selected),
            given: self.node_name_list(&kernel.context),
            expression: IdentificationExpr::Product(factors),
        });
        kernel
    }

    fn formula(&self, outcomes: &[usize], treatments: &[usize], expression: IdentificationExpr) -> IdentificationFormula {
        // Recursive q-kernels can carry external contexts that are not part of
        // the requested query. Close those contexts using their joint observed
        // marginal so the final formula has no free variables.
        let assigned: HashSet<String> = outcomes.iter().chain(treatments.iter())
            .map(|&node| self.names[node].clone()).collect();
        let mut free_context = HashSet::new();
        let mut visiting = HashSet::new();
        collect_free_contexts(
            &expression,
            &HashSet::new(),
            &assigned,
            &self.definitions,
            &mut visiting,
            &mut free_context,
        );
        let free_nodes: Vec<usize> = self.names.iter().enumerate()
            .filter_map(|(node, name)| free_context.contains(name).then_some(node)).collect();
        let expression = if free_nodes.is_empty() {
            expression
        } else {
            let variables = self.node_name_list(&free_nodes);
            IdentificationExpr::Sum {
                variables: variables.clone(),
                expression: Box::new(IdentificationExpr::Product(vec![
                    self.kernel_expr("P", &free_nodes, &[]),
                    expression,
                ])),
            }
        };
        IdentificationFormula {
            outcomes: self.node_name_list(outcomes),
            treatments: self.node_name_list(treatments),
            conditions: vec![],
            definitions: self.definitions.clone(),
            expression,
        }
    }
}

fn collect_free_contexts(
    expression: &IdentificationExpr,
    bound: &HashSet<String>,
    assigned: &HashSet<String>,
    definitions: &[IdentificationDefinition],
    visiting: &mut HashSet<String>,
    free: &mut HashSet<String>,
) {
    let add_context = |variables: &[String], bound: &HashSet<String>, assigned: &HashSet<String>, free: &mut HashSet<String>| {
        free.extend(variables.iter().filter(|variable| {
            !bound.contains(*variable) && !assigned.contains(*variable)
        }).cloned());
    };
    match expression {
        IdentificationExpr::Kernel { name, given, .. } => {
            add_context(given, bound, assigned, free);
            if name != "P" && visiting.insert(name.clone()) {
                if let Some(definition) = definitions.iter().find(|definition| definition.name == *name) {
                    let mut nested_bound = bound.clone();
                    nested_bound.extend(definition.variables.iter().cloned());
                    add_context(&definition.given, &nested_bound, assigned, free);
                    collect_free_contexts(
                        &definition.expression,
                        &nested_bound,
                        assigned,
                        definitions,
                        visiting,
                        free,
                    );
                }
                visiting.remove(name);
            }
        }
        IdentificationExpr::Sum { variables, expression } => {
            let mut nested_bound = bound.clone();
            nested_bound.extend(variables.iter().cloned());
            collect_free_contexts(expression, &nested_bound, assigned, definitions, visiting, free);
        }
        IdentificationExpr::Product(factors) => {
            for factor in factors {
                collect_free_contexts(factor, bound, assigned, definitions, visiting, free);
            }
        }
        IdentificationExpr::Quotient { numerator, denominator } => {
            collect_free_contexts(numerator, bound, assigned, definitions, visiting, free);
            collect_free_contexts(denominator, bound, assigned, definitions, visiting, free);
        }
    }
}

fn id_recursive(
    context: &mut IdContext<'_>,
    graph: &IdGraph,
    outcomes: &HashSet<usize>,
    treatments: &HashSet<usize>,
    kernel: &IdKernel,
) -> Result<IdentificationExpr, IdFailure> {
    let nodes = &graph.nodes;
    let node_set: HashSet<usize> = nodes.iter().copied().collect();
    if !outcomes.is_subset(&node_set)
        || !treatments.is_subset(&node_set)
        || kernel.scope.as_slice() != nodes.as_slice()
    {
        return Err(IdFailure::Invalid("Internal ID recursion received inconsistent graph and kernel scopes".into()));
    }

    // ID 1: no intervention remains, so return the requested marginal of the current kernel.
    if treatments.is_empty() {
        let keep: Vec<usize> = nodes.iter().copied().filter(|node| outcomes.contains(node)).collect();
        let eliminated: Vec<usize> = nodes.iter().copied().filter(|node| !outcomes.contains(node)).collect();
        let expression = if eliminated.is_empty() {
            context.kernel_expr(&kernel.name, &keep, &kernel.context)
        } else {
            IdentificationExpr::Sum {
                variables: context.node_name_list(&eliminated),
                expression: Box::new(context.kernel_expr(&kernel.name, nodes, &kernel.context)),
            }
        };
        return Ok(expression);
    }

    // ID 2: restrict to ancestors of the outcomes and marginalize the current kernel.
    let ancestors = graph.ancestors(&outcomes.iter().copied().collect::<Vec<_>>(), &HashSet::new());
    if ancestors.len() < nodes.len() {
        let ancestor_nodes: Vec<usize> = nodes.iter().copied().filter(|node| ancestors.contains(node)).collect();
        let reduced_graph = graph.induced(&ancestors);
        let reduced_kernel = context.marginal(kernel, &ancestor_nodes);
        let reduced_treatments: HashSet<usize> = treatments.intersection(&ancestors).copied().collect();
        return id_recursive(context, &reduced_graph, outcomes, &reduced_treatments, &reduced_kernel);
    }

    // ID 3: add variables that cease to be ancestors after incoming treatment arrows are cut.
    let ancestors_after_cut = graph.ancestors(
        &outcomes.iter().copied().collect::<Vec<_>>(),
        treatments,
    );
    let extra_treatments: HashSet<usize> = nodes.iter().copied()
        .filter(|node| !treatments.contains(node) && !ancestors_after_cut.contains(node))
        .collect();
    if !extra_treatments.is_empty() {
        let expanded_treatments: HashSet<usize> = treatments.union(&extra_treatments).copied().collect();
        return id_recursive(context, graph, outcomes, &expanded_treatments, kernel);
    }

    // ID 4: split the graph after deleting treatment nodes and recurse over each district.
    let remaining: HashSet<usize> = node_set.difference(treatments).copied().collect();
    let remaining_graph = graph.induced(&remaining);
    let remaining_components = remaining_graph.c_components();
    if remaining_components.len() > 1 {
        let mut component_formulas = Vec::new();
        for component in remaining_components {
            let component_set: HashSet<usize> = component.iter().copied().collect();
            let component_treatments: HashSet<usize> = node_set.difference(&component_set).copied().collect();
            component_formulas.push(id_recursive(
                context,
                graph,
                &component_set,
                &component_treatments,
                kernel,
            )?);
        }
        let summed_nodes: Vec<usize> = nodes.iter().copied()
            .filter(|node| !outcomes.contains(node) && !treatments.contains(node))
            .collect();
        let product = IdentificationExpr::Product(component_formulas);
        return Ok(if summed_nodes.is_empty() {
            product
        } else {
            IdentificationExpr::Sum {
                variables: context.node_name_list(&summed_nodes),
                expression: Box::new(product),
            }
        });
    }

    let inner = remaining_components.into_iter().next().ok_or_else(|| {
        IdFailure::Invalid("ID reached an empty post-intervention graph unexpectedly".into())
    })?;
    let inner_set: HashSet<usize> = inner.iter().copied().collect();
    let graph_components = graph.c_components();

    // ID 5: a single district before and after treatment deletion is a hedge witness.
    if graph_components.len() == 1 && inner_set.len() < nodes.len() {
        return Err(IdFailure::Hedge { outer: nodes.clone(), inner });
    }

    let order = graph.topological_order().ok_or_else(|| {
        IdFailure::Invalid("ID requires an acyclic directed graph; the causal graph contains a directed cycle".into())
    })?;

    // ID 6: the remaining district is already a district of G; extract its q-factor.
    if graph_components.iter().any(|component| component.iter().copied().collect::<HashSet<_>>() == inner_set) {
        let mut factors = Vec::new();
        let mut preceding = Vec::new();
        for &node in &order {
            if inner_set.contains(&node) {
                factors.push(context.conditional_expr(kernel, node, &preceding));
            }
            preceding.push(node);
        }
        let summed_nodes: Vec<usize> = inner.iter().copied().filter(|node| !outcomes.contains(node)).collect();
        let product = IdentificationExpr::Product(factors);
        return Ok(if summed_nodes.is_empty() {
            product
        } else {
            IdentificationExpr::Sum {
                variables: context.node_name_list(&summed_nodes),
                expression: Box::new(product),
            }
        });
    }

    // ID 7: obtain Q[S'] for the containing district and recurse on its induced graph.
    let containing_district = graph_components.into_iter()
        .find(|component| inner.iter().all(|node| component.contains(node)))
        .ok_or_else(|| IdFailure::Invalid("Could not locate the district containing the post-intervention district".into()))?;
    let district_set: HashSet<usize> = containing_district.iter().copied().collect();
    let q_kernel = context.q_factor(kernel, &containing_district, &order);
    let district_graph = graph.induced(&district_set);
    let district_treatments: HashSet<usize> = treatments.intersection(&district_set).copied().collect();
    id_recursive(context, &district_graph, outcomes, &district_treatments, &q_kernel)
}

impl CausalModel {
    pub fn new(
        names: Vec<String>,
        weights: Vec<Vec<f64>>,
        noise_variances: Vec<f64>,
        noise_means: Vec<f64>,
    ) -> Self {
        let mut name_to_idx = HashMap::new();
        for (i, name) in names.iter().enumerate() {
            name_to_idx.insert(name.clone(), i);
        }
        Self {
            names,
            name_to_idx,
            weights,
            noise_variances,
            noise_means,
            noise_covariances: HashMap::new(),
            bidirected: Vec::new(),
        }
    }

    fn build_noise_covariance_matrix(
        &self,
        off_diagonal: &HashMap<(usize, usize), f64>,
    ) -> Option<Vec<Vec<f64>>> {
        let n = self.names.len();
        if self.noise_variances.len() != n || self.noise_means.len() != n {
            return None;
        }

        let mut covariance = vec![vec![0.0; n]; n];
        for i in 0..n {
            let variance = self.noise_variances[i];
            if !variance.is_finite() || variance < 0.0 {
                return None;
            }
            covariance[i][i] = variance;
        }
        for (&(i, j), &value) in off_diagonal {
            if i >= j || j >= n || !value.is_finite() {
                return None;
            }
            covariance[i][j] = value;
            covariance[j][i] = value;
        }

        if is_positive_semidefinite(&covariance) {
            Some(covariance)
        } else {
            None
        }
    }

    /// Helper to compute (I - W^T)^-1
    fn get_inv_i_minus_wt(&self) -> Option<Vec<Vec<f64>>> {
        let n = self.names.len();
        let mut mat = vec![vec![0.0; n]; n];
        for i in 0..n {
            mat[i][i] = 1.0;
            for j in 0..n {
                // transpose: W^T is weights[j][i]
                mat[i][j] -= self.weights[j][i];
            }
        }
        invert_matrix(&mat)
    }

    /// Compute the joint mean vector and covariance matrix of X.
    /// X = (I - W^T)^-1 * U
    pub fn compute_joint_distribution(&self) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
        let n = self.names.len();
        let inv = self.get_inv_i_minus_wt()?;
        let noise_covariance = self.build_noise_covariance_matrix(&self.noise_covariances)?;

        // Mean: mu_X = inv * mu_U
        let mut mean = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                mean[i] += inv[i][j] * self.noise_means[j];
            }
        }

        // Covariance: Sigma_X = inv * Sigma_U * inv^T. Sigma_U includes
        // off-diagonal entries for explicitly weighted bidirected edges.
        let mut inv_times_noise = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    inv_times_noise[i][j] += inv[i][k] * noise_covariance[k][j];
                }
            }
        }
        let mut cov = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += inv_times_noise[i][k] * inv[j][k];
                }
                cov[i][j] = sum;
            }
        }

        Some((mean, cov))
    }

    /// Condition via Bayes Rule: observe(var, val)
    /// Returns the conditional mean and standard deviation for all variables.
    pub fn observe(&self, evidence: &HashMap<String, f64>) -> Option<HashMap<String, (f64, f64)>> {
        let (mu, sigma) = self.compute_joint_distribution()?;
        let n = self.names.len();

        let mut evidence_indices = Vec::new();
        let mut evidence_values = Vec::new();
        for (name, &val) in evidence {
            if let Some(&idx) = self.name_to_idx.get(name) {
                evidence_indices.push(idx);
                evidence_values.push(val);
            }
        }

        if evidence_indices.is_empty() {
            // No evidence, return joint stats
            let mut results = HashMap::new();
            for i in 0..n {
                let std = sigma[i][i].sqrt();
                results.insert(self.names[i].clone(), (mu[i], std));
            }
            return Some(results);
        }

        // Partition into Query (Q) and Evidence (E)
        let mut q_indices = Vec::new();
        for i in 0..n {
            if !evidence_indices.contains(&i) {
                q_indices.push(i);
            }
        }

        // Submatrices
        // Sigma_EE
        let num_e = evidence_indices.len();
        let mut sig_ee = vec![vec![0.0; num_e]; num_e];
        for i in 0..num_e {
            for j in 0..num_e {
                sig_ee[i][j] = sigma[evidence_indices[i]][evidence_indices[j]];
            }
        }
        let sig_ee_inv = invert_matrix(&sig_ee)?;

        // Conditional mean and var for each query variable
        let mut results = HashMap::new();
        for &q in &q_indices {
            // Sigma_qE: 1 x num_e
            let mut sig_qe = vec![0.0; num_e];
            for i in 0..num_e {
                sig_qe[i] = sigma[q][evidence_indices[i]];
            }

            // sig_qe * sig_ee_inv
            let mut coeff = vec![0.0; num_e];
            for i in 0..num_e {
                for j in 0..num_e {
                    coeff[i] += sig_qe[j] * sig_ee_inv[j][i];
                }
            }

            // mu_q|E = mu_q + coeff * (x_E - mu_E)
            let mut cond_mean = mu[q];
            for i in 0..num_e {
                cond_mean += coeff[i] * (evidence_values[i] - mu[evidence_indices[i]]);
            }

            // var_q|E = Sigma_qq - sig_qe * sig_ee_inv * sig_Eq
            let mut cond_var = sigma[q][q];
            let mut sum_deduction = 0.0;
            for i in 0..num_e {
                sum_deduction += coeff[i] * sigma[evidence_indices[i]][q];
            }
            cond_var -= sum_deduction;
            let cond_std = cond_var.max(0.0).sqrt();

            results.insert(self.names[q].clone(), (cond_mean, cond_std));
        }

        // Include evidence variables as deterministic results
        for (name, &val) in evidence {
            results.insert(name.clone(), (val, 0.0));
        }

        Some(results)
    }

    /// Intervene: do(var = val) cuts incoming edges and propagates values forward.
    pub fn intervene(&self, intervention: &HashMap<String, f64>) -> Option<HashMap<String, (f64, f64)>> {
        let n = self.names.len();
        // Create an intervened model
        let mut int_weights = self.weights.clone();
        let mut int_noise_vars = self.noise_variances.clone();
        let mut int_noise_means = self.noise_means.clone();
        let mut intervened_indices = HashSet::new();

        for (name, &val) in intervention {
            if let Some(&idx) = self.name_to_idx.get(name) {
                intervened_indices.insert(idx);
                // Cut incoming edges: weights[*][idx] = 0
                for r in 0..n {
                    int_weights[r][idx] = 0.0;
                }
                // Set structural equation of idx to be deterministic: X_idx = val + 0*U
                int_noise_means[idx] = val;
                int_noise_vars[idx] = 1e-15; // essentially zero variance
            }
        }

        let mut int_model = CausalModel::new(
            self.names.clone(),
            int_weights,
            int_noise_vars,
            int_noise_means,
        );
        // An ideal intervention removes the intervened variables' exogenous
        // noise and any shared-noise edges incident to them. Preserve the
        // remaining latent covariances among non-intervened variables.
        int_model.bidirected = self.bidirected.iter().copied()
            .filter(|(u, v)| !intervened_indices.contains(u) && !intervened_indices.contains(v))
            .collect();
        int_model.noise_covariances = self.noise_covariances.iter()
            .filter(|((u, v), _)| !intervened_indices.contains(u) && !intervened_indices.contains(v))
            .map(|(pair, covariance)| (*pair, *covariance))
            .collect();
        int_model.compute_joint_distribution().map(|(mu, sigma)| {
            let mut results = HashMap::new();
            for i in 0..n {
                let std = sigma[i][i].sqrt();
                results.insert(self.names[i].clone(), (mu[i], std));
            }
            results
        })
    }

    /// Counterfactual Logic: Abduction, Action, Prediction
    pub fn counterfactual(
        &self,
        evidence: &HashMap<String, f64>,
        intervention: &HashMap<String, f64>,
        query: &[String],
    ) -> Option<HashMap<String, f64>> {
        let n = self.names.len();
        // Step 1: Abduction. Infer exogenous noise values U given evidence.
        // We know U = (I - W^T) X.
        // We want to compute the conditional expectation of U given X_E = x_E.
        // Let's find the joint distribution of [U; X] which is Gaussian.
        // Mean of U is self.noise_means. Cov(U, X) = Sigma_U * inv^T,
        // where inv = (I - W^T)^-1.
        let inv = self.get_inv_i_minus_wt()?;
        let (mu_x, sigma_x) = self.compute_joint_distribution()?;
        let noise_covariance = self.build_noise_covariance_matrix(&self.noise_covariances)?;

        let mut evidence_indices = Vec::new();
        let mut evidence_values = Vec::new();
        for (name, &val) in evidence {
            if let Some(&idx) = self.name_to_idx.get(name) {
                evidence_indices.push(idx);
                evidence_values.push(val);
            }
        }

        let u_posterior = if evidence_indices.is_empty() {
            self.noise_means.clone()
        } else {
            // We use conditional expectation: E[U | X_E = x_E]
            // E[U | X_E] = mu_U + Cov(U, X_E) * Cov(X_E, X_E)^-1 * (x_E - mu_E)
            let num_e = evidence_indices.len();
            // Cov(U, X_E) is an n x num_e matrix.
            // Cov(U_i, X_e) = sum_k Sigma_U[i][k] * inv[e][k].
            let mut cov_u_xe = vec![vec![0.0; num_e]; n];
            for i in 0..n {
                for (j_idx, &e_col) in evidence_indices.iter().enumerate() {
                    for k in 0..n {
                        cov_u_xe[i][j_idx] += noise_covariance[i][k] * inv[e_col][k];
                    }
                }
            }

            // Cov(X_E, X_E)^-1
            let mut sig_ee = vec![vec![0.0; num_e]; num_e];
            for i in 0..num_e {
                for j in 0..num_e {
                    sig_ee[i][j] = sigma_x[evidence_indices[i]][evidence_indices[j]];
                }
            }
            let sig_ee_inv = invert_matrix(&sig_ee)?;

            // E[U | X_E] computation
            let mut u_mean = self.noise_means.clone();
            for i in 0..n {
                // Row i of cov_u_xe multiplied by sig_ee_inv
                let mut coeff = vec![0.0; num_e];
                for j in 0..num_e {
                    for k in 0..num_e {
                        coeff[j] += cov_u_xe[i][k] * sig_ee_inv[k][j];
                    }
                }

                let mut diff = 0.0;
                for j in 0..num_e {
                    diff += coeff[j] * (evidence_values[j] - mu_x[evidence_indices[j]]);
                }
                u_mean[i] += diff;
            }
            u_mean
        };

        // Step 2 & 3: Action & Prediction.
        // Intervene on the structural equations by setting X_I = x_I.
        // And propagate the posterior noise values u_posterior.
        // Solve (I - W'^T) X = U', where U'_i = x_i if i in I, and U'_i = u_posterior_i if i not in I.
        // Also W'_ji = W_ji if i not in I, and 0 if i in I.
        let mut int_weights = self.weights.clone();
        let mut u_prime = u_posterior.clone();

        for (name, &val) in intervention {
            if let Some(&idx) = self.name_to_idx.get(name) {
                for r in 0..n {
                    int_weights[r][idx] = 0.0;
                }
                u_prime[idx] = val;
            }
        }

        // Solve: X_i = sum_j (int_weights[j][i] * X_j) + u_prime[i]
        // This is (I - W'^T) X = U'. So X = (I - W'^T)^-1 * U'.
        let mut mat = vec![vec![0.0; n]; n];
        for i in 0..n {
            mat[i][i] = 1.0;
            for j in 0..n {
                mat[i][j] -= int_weights[j][i];
            }
        }
        let int_inv = invert_matrix(&mat)?;

        let mut x_counterfactual = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                x_counterfactual[i] += int_inv[i][j] * u_prime[j];
            }
        }

        let mut results = HashMap::new();
        for name in query {
            if let Some(&idx) = self.name_to_idx.get(name) {
                results.insert(name.clone(), x_counterfactual[idx]);
            }
        }
        Some(results)
    }

    /// Add a bi-directed / confounded edge u <-> v
    pub fn add_bidirected(&mut self, u: &str, v: &str) {
        if let (Some(&i), Some(&j)) = (self.name_to_idx.get(u), self.name_to_idx.get(v)) {
            let pair = if i < j { (i, j) } else { (j, i) };
            if !self.bidirected.contains(&pair) {
                self.bidirected.push(pair);
            }
        }
    }

    /// Add a bidirected edge and set the covariance between its exogenous noises.
    /// The full exogenous covariance matrix must remain positive semidefinite.
    pub fn add_bidirected_with_covariance(
        &mut self,
        u: &str,
        v: &str,
        covariance: f64,
    ) -> Result<(), String> {
        let (&i, &j) = match (self.name_to_idx.get(u), self.name_to_idx.get(v)) {
            (Some(i), Some(j)) if i != j => (i, j),
            (Some(_), Some(_)) => return Err("A bidirected covariance requires two distinct variables".into()),
            _ => return Err(format!("Unknown variable in bidirected edge '{} <-> {}'", u, v)),
        };
        if !covariance.is_finite() {
            return Err("Bidirected covariance must be finite".into());
        }

        let pair = if i < j { (i, j) } else { (j, i) };
        let mut candidate = self.noise_covariances.clone();
        candidate.insert(pair, covariance);
        if self.build_noise_covariance_matrix(&candidate).is_none() {
            return Err(format!(
                "Bidirected covariance for '{} <-> {}' would make the exogenous noise covariance matrix invalid",
                u, v
            ));
        }

        self.add_bidirected(u, v);
        self.noise_covariances = candidate;
        Ok(())
    }

    pub fn parents(&self, node: usize) -> Vec<usize> {
        let mut p = Vec::new();
        for i in 0..self.names.len() {
            if self.weights[i][node].abs() > 1e-12 {
                p.push(i);
            }
        }
        p
    }

    pub fn children(&self, node: usize) -> Vec<usize> {
        let mut c = Vec::new();
        for j in 0..self.names.len() {
            if self.weights[node][j].abs() > 1e-12 {
                c.push(j);
            }
        }
        c
    }

    pub fn ancestors(&self, nodes: &[usize]) -> HashSet<usize> {
        let mut anc = HashSet::new();
        let mut queue: VecDeque<usize> = nodes.iter().cloned().collect();
        while let Some(curr) = queue.pop_front() {
            if anc.insert(curr) {
                for p in self.parents(curr) {
                    queue.push_back(p);
                }
            }
        }
        anc
    }

    pub fn descendants(&self, nodes: &[usize]) -> HashSet<usize> {
        let mut desc = HashSet::new();
        let mut queue: VecDeque<usize> = nodes.iter().cloned().collect();
        while let Some(curr) = queue.pop_front() {
            if desc.insert(curr) {
                for c in self.children(curr) {
                    queue.push_back(c);
                }
            }
        }
        desc
    }

    pub fn c_components(&self, subset: &HashSet<usize>) -> Vec<HashSet<usize>> {
        let mut visited = HashSet::new();
        let mut components = Vec::new();

        for &node in subset {
            if visited.contains(&node) {
                continue;
            }
            let mut comp = HashSet::new();
            let mut queue = VecDeque::new();
            queue.push_back(node);
            visited.insert(node);
            comp.insert(node);

            while let Some(curr) = queue.pop_front() {
                for &(u, v) in &self.bidirected {
                    let neighbor = if u == curr && subset.contains(&v) {
                        Some(v)
                    } else if v == curr && subset.contains(&u) {
                        Some(u)
                    } else {
                        None
                    };
                    if let Some(n) = neighbor {
                        if visited.insert(n) {
                            comp.insert(n);
                            queue.push_back(n);
                        }
                    }
                }
            }
            components.push(comp);
        }
        components
    }

    /// Check d-separation (m-separation) between X and Y given Z in an ADMG.
    pub fn is_d_separated(&self, x: &[usize], y: &[usize], z: &[usize], cut_outgoing: Option<&[usize]>) -> bool {
        let n = self.names.len();
        let z_set: HashSet<usize> = z.iter().cloned().collect();
        let x_set: HashSet<usize> = x.iter().cloned().collect();
        let y_set: HashSet<usize> = y.iter().cloned().collect();
        let cut_set: HashSet<usize> = cut_outgoing.map(|c| c.iter().cloned().collect()).unwrap_or_default();

        let mut desc_map: Vec<HashSet<usize>> = Vec::with_capacity(n);
        for i in 0..n {
            desc_map.push(self.descendants(&[i]));
        }

        for &start in &x_set {
            let mut visited = vec![false; n];
            visited[start] = true;
            let mut path = vec![start];
            let mut edges = vec![];

            if self.find_active_trail(start, &x_set, &y_set, &z_set, &cut_set, &desc_map, &mut visited, &mut path, &mut edges) {
                return false;
            }
        }
        true
    }

    fn find_active_trail(
        &self,
        curr: usize,
        x_set: &HashSet<usize>,
        y_set: &HashSet<usize>,
        z_set: &HashSet<usize>,
        cut_set: &HashSet<usize>,
        desc_map: &[HashSet<usize>],
        visited: &mut [bool],
        path: &mut Vec<usize>,
        edges: &mut Vec<(bool, bool)>,
    ) -> bool {
        if path.len() > 1 && y_set.contains(&curr) {
            if self.is_trail_active(path, edges, z_set, desc_map) {
                return true;
            }
        }

        let n = self.names.len();
        let mut neighbors = Vec::new();

        // 1. Directed edges from curr -> j
        if !cut_set.contains(&curr) {
            for j in 0..n {
                if self.weights[curr][j].abs() > 1e-12 && !visited[j] {
                    neighbors.push((j, false, true));
                }
            }
        }

        // 2. Directed edges from j -> curr
        for j in 0..n {
            if self.weights[j][curr].abs() > 1e-12 && !cut_set.contains(&j) && !visited[j] {
                neighbors.push((j, true, false));
            }
        }

        // 3. Bidirected edges between curr and j
        for &(u, v) in &self.bidirected {
            let j_opt = if u == curr && !visited[v] {
                Some(v)
            } else if v == curr && !visited[u] {
                Some(u)
            } else {
                None
            };
            if let Some(j) = j_opt {
                neighbors.push((j, true, true));
            }
        }

        for (next, in_curr, in_next) in neighbors {
            visited[next] = true;
            path.push(next);
            edges.push((in_curr, in_next));

            if path.len() < 3 || self.is_internal_node_active(path.len() - 2, path, edges, z_set, desc_map) {
                if self.find_active_trail(next, x_set, y_set, z_set, cut_set, desc_map, visited, path, edges) {
                    return true;
                }
            }

            edges.pop();
            path.pop();
            visited[next] = false;
        }

        false
    }

    fn is_internal_node_active(
        &self,
        i: usize,
        path: &[usize],
        edges: &[(bool, bool)],
        z_set: &HashSet<usize>,
        desc_map: &[HashSet<usize>],
    ) -> bool {
        let node = path[i];
        let (_, in_to_curr_from_prev) = edges[i - 1];
        let (in_to_curr_from_next, _) = edges[i];

        let is_collider = in_to_curr_from_prev && in_to_curr_from_next;

        if is_collider {
            z_set.contains(&node) || desc_map[node].iter().any(|d| z_set.contains(d))
        } else {
            !z_set.contains(&node)
        }
    }

    fn is_trail_active(
        &self,
        path: &[usize],
        edges: &[(bool, bool)],
        z_set: &HashSet<usize>,
        desc_map: &[HashSet<usize>],
    ) -> bool {
        for i in 1..path.len() - 1 {
            if !self.is_internal_node_active(i, path, edges, z_set, desc_map) {
                return false;
            }
        }
        true
    }

    /// Discover minimal admissible Backdoor adjustment set Z for treatment X and outcome Y
    pub fn find_backdoor_admissible(&self, treatment: &str, outcome: &str) -> Option<Vec<String>> {
        let &x = self.name_to_idx.get(treatment)?;
        let &y = self.name_to_idx.get(outcome)?;
        let n = self.names.len();

        let desc_x = self.descendants(&[x]);
        let mut candidates = Vec::new();
        for i in 0..n {
            if i != x && i != y && !desc_x.contains(&i) {
                candidates.push(i);
            }
        }

        if self.is_d_separated(&[x], &[y], &[], Some(&[x])) {
            return Some(vec![]);
        }

        let parents_x: Vec<usize> = self.parents(x).into_iter().filter(|p| candidates.contains(p)).collect();
        if !parents_x.is_empty() && self.is_d_separated(&[x], &[y], &parents_x, Some(&[x])) {
            return Some(parents_x.iter().map(|&i| self.names[i].clone()).collect());
        }

        let max_k = candidates.len().min(4);
        for k in 1..=max_k {
            for combo in get_combinations(&candidates, k) {
                if self.is_d_separated(&[x], &[y], &combo, Some(&[x])) {
                    return Some(combo.iter().map(|&i| self.names[i].clone()).collect());
                }
            }
        }

        None
    }

    /// Discover Frontdoor adjustment mediator set M for treatment X and outcome Y
    pub fn find_frontdoor_admissible(&self, treatment: &str, outcome: &str) -> Option<Vec<String>> {
        let &x = self.name_to_idx.get(treatment)?;
        let &y = self.name_to_idx.get(outcome)?;
        let n = self.names.len();

        let desc_x = self.descendants(&[x]);
        let anc_y = self.ancestors(&[y]);

        let candidates: Vec<usize> = (0..n)
            .filter(|&i| i != x && i != y && desc_x.contains(&i) && anc_y.contains(&i))
            .collect();

        if candidates.is_empty() {
            return None;
        }

        for k in 1..=candidates.len().min(3) {
            for combo in get_combinations(&candidates, k) {
                let m_set: Vec<usize> = combo;

                if !self.has_directed_path_excluding(x, y, &m_set) {
                    if self.is_d_separated(&[x], &m_set, &[], Some(&[x])) {
                        if self.is_d_separated(&m_set, &[y], &[x], Some(&m_set)) {
                            return Some(m_set.iter().map(|&i| self.names[i].clone()).collect());
                        }
                    }
                }
            }
        }
        None
    }

    fn has_directed_path_excluding(&self, start: usize, target: usize, excluded: &[usize]) -> bool {
        let excl_set: HashSet<usize> = excluded.iter().cloned().collect();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited.insert(start);

        while let Some(curr) = queue.pop_front() {
            if curr == target {
                return true;
            }
            for next in self.children(curr) {
                if !excl_set.contains(&next) && visited.insert(next) {
                    queue.push_back(next);
                }
            }
        }
        false
    }

    /// Complete Causal Identification Algorithm (Tian & Pearl 2002, Shpitser & Pearl 2006)
    pub fn identify(&self, query_outcomes: &[String], treatments: &[String]) -> IdentificationResult {
        if query_outcomes.is_empty() || treatments.is_empty() {
            return IdentificationResult {
                identifiable: false,
                method: "Invalid".into(),
                formula: "Empty query or treatments".into(),
                adjustment_set: vec![],
                hedge: None,
                hedge_outer: None,
                hedge_inner: None,
                formula_ast: None,
            };
        }
        let invalid = |message: String| IdentificationResult {
            identifiable: false,
            method: "Invalid".into(),
            formula: message,
            adjustment_set: vec![],
            hedge: None,
            hedge_outer: None,
            hedge_inner: None,
            formula_ast: None,
        };

        let graph = match IdGraph::from_model(self) {
            Ok(graph) => graph,
            Err(error) => return invalid(error),
        };
        if graph.topological_order().is_none() {
            return invalid("ID requires an acyclic directed graph; the causal graph contains a directed cycle".into());
        }

        let resolve = |requested: &[String], kind: &str| -> Result<HashSet<usize>, String> {
            let mut resolved = HashSet::new();
            for name in requested {
                let Some(&index) = self.name_to_idx.get(name) else {
                    return Err(format!("Unknown {} '{}'", kind, name));
                };
                if index >= self.names.len() || self.names[index] != *name {
                    return Err(format!("Invalid variable index for {} '{}'", kind, name));
                }
                resolved.insert(index);
            }
            Ok(resolved)
        };
        let outcome_set = match resolve(query_outcomes, "outcome") {
            Ok(nodes) => nodes,
            Err(message) => return invalid(message),
        };
        let treatment_set = match resolve(treatments, "treatment") {
            Ok(nodes) => nodes,
            Err(message) => return invalid(message),
        };
        if outcome_set.is_empty() || treatment_set.is_empty() {
            return invalid("Query and treatment sets must contain at least one known variable".into());
        }
        if !outcome_set.is_disjoint(&treatment_set) {
            return invalid("Outcome and treatment sets must be disjoint".into());
        }

        let outcome_nodes: Vec<usize> = graph.nodes.iter().copied().filter(|node| outcome_set.contains(node)).collect();
        let treatment_nodes: Vec<usize> = graph.nodes.iter().copied().filter(|node| treatment_set.contains(node)).collect();

        // Preserve the familiar closed-form labels when the corresponding sound criterion applies.
        if outcome_nodes.len() == 1 && treatment_nodes.len() == 1 {
            let y_name = &self.names[outcome_nodes[0]];
            let x_name = &self.names[treatment_nodes[0]];
            if let Some(z_names) = self.find_backdoor_admissible(x_name, y_name) {
                let expression = if z_names.is_empty() {
                    IdentificationExpr::Kernel {
                        name: "P".into(),
                        variables: vec![y_name.clone()],
                        given: vec![x_name.clone()],
                    }
                } else {
                    let mut conditioning = vec![x_name.clone()];
                    conditioning.extend(z_names.iter().cloned());
                    IdentificationExpr::Sum {
                        variables: z_names.clone(),
                        expression: Box::new(IdentificationExpr::Product(vec![
                            IdentificationExpr::Kernel {
                                name: "P".into(),
                                variables: vec![y_name.clone()],
                                given: conditioning,
                            },
                            IdentificationExpr::Kernel {
                                name: "P".into(),
                                variables: z_names.clone(),
                                given: vec![],
                            },
                        ])),
                    }
                };
                let formula_ast = IdentificationFormula {
                    outcomes: vec![y_name.clone()],
                    treatments: vec![x_name.clone()],
                    conditions: vec![],
                    definitions: vec![],
                    expression,
                };
                return IdentificationResult {
                    identifiable: true,
                    method: "Backdoor".into(),
                    formula: formula_ast.render(),
                    adjustment_set: z_names,
                    hedge: None,
                    hedge_outer: None,
                    hedge_inner: None,
                    formula_ast: Some(formula_ast),
                };
            }

            if let Some(m_names) = self.find_frontdoor_admissible(x_name, y_name) {
                let inner = IdentificationExpr::Sum {
                    variables: vec![format!("{}'", x_name)],
                    expression: Box::new(IdentificationExpr::Product(vec![
                        IdentificationExpr::Kernel {
                            name: "P".into(),
                            variables: vec![y_name.clone()],
                            given: [vec![format!("{}'", x_name)], m_names.clone()].concat(),
                        },
                        IdentificationExpr::Kernel {
                            name: "P".into(),
                            variables: vec![format!("{}'", x_name)],
                            given: vec![],
                        },
                    ])),
                };
                let expression = IdentificationExpr::Sum {
                    variables: m_names.clone(),
                    expression: Box::new(IdentificationExpr::Product(vec![
                        IdentificationExpr::Kernel {
                            name: "P".into(),
                            variables: m_names.clone(),
                            given: vec![x_name.clone()],
                        },
                        inner,
                    ])),
                };
                let formula_ast = IdentificationFormula {
                    outcomes: vec![y_name.clone()],
                    treatments: vec![x_name.clone()],
                    conditions: vec![],
                    definitions: vec![],
                    expression,
                };
                return IdentificationResult {
                    identifiable: true,
                    method: "Frontdoor".into(),
                    formula: formula_ast.render(),
                    adjustment_set: m_names,
                    hedge: None,
                    hedge_outer: None,
                    hedge_inner: None,
                    formula_ast: Some(formula_ast),
                };
            }
        }

        let mut context = IdContext::new(&self.names);
        let source = IdKernel { name: "P".into(), scope: graph.nodes.clone(), context: vec![] };
        match id_recursive(&mut context, &graph, &outcome_set, &treatment_set, &source) {
            Ok(expression) => {
                let formula_ast = context.formula(&outcome_nodes, &treatment_nodes, expression);
                IdentificationResult {
                    identifiable: true,
                    method: "ID".into(),
                    formula: formula_ast.render(),
                    adjustment_set: vec![],
                    hedge: None,
                    hedge_outer: None,
                    hedge_inner: None,
                    formula_ast: Some(formula_ast),
                }
            }
            Err(IdFailure::Hedge { outer, inner }) => {
                let outer_names: Vec<String> = outer.iter().map(|&node| self.names[node].clone()).collect();
                let inner_names: Vec<String> = inner.iter().map(|&node| self.names[node].clone()).collect();
                let mut flattened = outer_names.clone();
                flattened.extend(inner_names.iter().cloned());
                flattened.sort();
                flattened.dedup();
                IdentificationResult {
                    identifiable: false,
                    method: "ID-Hedge".into(),
                    formula: format!(
                        "FAIL: ID returned hedge (F = {{{}}}, F' = {{{}}}); the effect is not identifiable from observational data",
                        outer_names.join(", "), inner_names.join(", ")
                    ),
                    adjustment_set: vec![],
                    hedge: Some(flattened),
                    hedge_outer: Some(outer_names),
                    hedge_inner: Some(inner_names),
                    formula_ast: None,
                }
            }
            Err(IdFailure::Invalid(message)) => invalid(message),
        }
    }

    /// Identify a conditional interventional query P(Y | do(X), Z) using IDC.
    pub fn identify_conditional(
        &self,
        query_outcomes: &[String],
        treatments: &[String],
        conditions: &[String],
    ) -> IdentificationResult {
        if conditions.is_empty() {
            return self.identify(query_outcomes, treatments);
        }
        let invalid = |message: String| IdentificationResult {
            identifiable: false,
            method: "Invalid".into(),
            formula: message,
            adjustment_set: vec![],
            hedge: None,
            hedge_outer: None,
            hedge_inner: None,
            formula_ast: None,
        };
        if query_outcomes.is_empty() || treatments.is_empty() {
            return invalid("Conditional identification requires non-empty outcomes and treatments".into());
        }
        let graph = match IdGraph::from_model(self) {
            Ok(graph) => graph,
            Err(error) => return invalid(error),
        };
        if graph.topological_order().is_none() {
            return invalid("IDC requires an acyclic directed graph; the causal graph contains a directed cycle".into());
        }
        let resolve = |requested: &[String], kind: &str| -> Result<HashSet<usize>, String> {
            let mut resolved = HashSet::new();
            for name in requested {
                let Some(&index) = self.name_to_idx.get(name) else {
                    return Err(format!("Unknown {} '{}'", kind, name));
                };
                if index >= self.names.len() || self.names[index] != *name {
                    return Err(format!("Invalid variable index for {} '{}'", kind, name));
                }
                resolved.insert(index);
            }
            Ok(resolved)
        };
        let outcome_set = match resolve(query_outcomes, "outcome") {
            Ok(nodes) if !nodes.is_empty() => nodes,
            Ok(_) => return invalid("Outcome set must contain at least one known variable".into()),
            Err(message) => return invalid(message),
        };
        let treatment_set = match resolve(treatments, "treatment") {
            Ok(nodes) if !nodes.is_empty() => nodes,
            Ok(_) => return invalid("Treatment set must contain at least one known variable".into()),
            Err(message) => return invalid(message),
        };
        let mut remaining_conditions = match resolve(conditions, "condition") {
            Ok(nodes) => nodes,
            Err(message) => return invalid(message),
        };
        if remaining_conditions.is_empty() {
            return self.identify(query_outcomes, treatments);
        }
        if !outcome_set.is_disjoint(&treatment_set)
            || !outcome_set.is_disjoint(&remaining_conditions)
            || !treatment_set.is_disjoint(&remaining_conditions)
        {
            return invalid("Outcomes, treatments, and conditions must be pairwise disjoint".into());
        }

        let outcome_nodes: Vec<usize> = graph.nodes.iter().copied()
            .filter(|node| outcome_set.contains(node)).collect();
        let treatment_nodes: Vec<usize> = graph.nodes.iter().copied()
            .filter(|node| treatment_set.contains(node)).collect();
        let condition_nodes: Vec<usize> = graph.nodes.iter().copied()
            .filter(|node| remaining_conditions.contains(node)).collect();
        let mut expanded_treatments = treatment_set.clone();

        // IDC Step 1: exchange any condition that is independent of the outcomes
        // in G_{bar X, underlined{Z_i}} into the intervention set.
        loop {
            let candidate_order: Vec<usize> = graph.nodes.iter().copied()
                .filter(|node| remaining_conditions.contains(node)).collect();
            let mut exchanged = None;
            for candidate in candidate_order {
                let mutilated = graph.mutilated(&expanded_treatments, &HashSet::from([candidate]));
                let mut given: Vec<usize> = expanded_treatments.iter().copied().collect();
                given.extend(remaining_conditions.iter().copied().filter(|node| *node != candidate));
                given.sort_unstable();
                given.dedup();
                if mutilated.m_separated(&outcome_nodes, &[candidate], &given) {
                    exchanged = Some(candidate);
                    break;
                }
            }
            let Some(candidate) = exchanged else { break };
            remaining_conditions.remove(&candidate);
            expanded_treatments.insert(candidate);
        }

        let expanded_treatment_nodes: Vec<usize> = graph.nodes.iter().copied()
            .filter(|node| expanded_treatments.contains(node)).collect();
        let remaining_condition_nodes: Vec<usize> = graph.nodes.iter().copied()
            .filter(|node| remaining_conditions.contains(node)).collect();
        let mut id_result = if remaining_condition_nodes.is_empty() {
            let expanded_names: Vec<String> = expanded_treatment_nodes.iter()
                .map(|&node| self.names[node].clone()).collect();
            self.identify(query_outcomes, &expanded_names)
        } else {
            // IDC Step 2: ID the joint distribution P(Y, Z | do(X)), then
            // normalize by its marginal over Y.
            let joint_nodes: Vec<usize> = graph.nodes.iter().copied()
                .filter(|node| outcome_set.contains(node) || remaining_conditions.contains(node)).collect();
            let joint_names: Vec<String> = joint_nodes.iter()
                .map(|&node| self.names[node].clone()).collect();
            let expanded_names: Vec<String> = expanded_treatment_nodes.iter()
                .map(|&node| self.names[node].clone()).collect();
            self.identify(&joint_names, &expanded_names)
        };

        if !id_result.identifiable {
            return id_result;
        }
        let Some(mut identified_formula) = id_result.formula_ast.take() else {
            return invalid("ID succeeded without producing a structured formula".into());
        };
        let expression = if remaining_condition_nodes.is_empty() {
            identified_formula.expression
        } else {
            let numerator = identified_formula.expression;
            let denominator = IdentificationExpr::Sum {
                variables: outcome_nodes.iter().map(|&node| self.names[node].clone()).collect(),
                expression: Box::new(numerator.clone()),
            };
            IdentificationExpr::Quotient {
                numerator: Box::new(numerator),
                denominator: Box::new(denominator),
            }
        };
        identified_formula.outcomes = outcome_nodes.iter().map(|&node| self.names[node].clone()).collect();
        identified_formula.treatments = treatment_nodes.iter().map(|&node| self.names[node].clone()).collect();
        identified_formula.conditions = condition_nodes.iter().map(|&node| self.names[node].clone()).collect();
        identified_formula.expression = expression;
        id_result.method = "IDC".into();
        id_result.formula = identified_formula.render();
        id_result.formula_ast = Some(identified_formula);
        id_result
    }

    /// Identify a categorical query and evaluate its formula against observed rows.
    pub fn estimate_distribution(
        &self,
        dataset: &CausalDataset,
        query_outcomes: &[String],
        interventions: &HashMap<String, String>,
        conditions: &HashMap<String, String>,
    ) -> Result<CausalEstimate, String> {
        if interventions.is_empty() {
            return Err("Categorical causal estimation requires at least one intervention value".into());
        }
        let treatment_names: Vec<String> = self.names.iter()
            .filter(|name| interventions.contains_key(*name)).cloned().collect();
        if treatment_names.len() != interventions.len() {
            let unknown = interventions.keys().find(|name| !self.name_to_idx.contains_key(*name))
                .cloned().unwrap_or_default();
            return Err(format!("Unknown intervention variable '{}'", unknown));
        }
        let condition_names: Vec<String> = self.names.iter()
            .filter(|name| conditions.contains_key(*name)).cloned().collect();
        if condition_names.len() != conditions.len() {
            let unknown = conditions.keys().find(|name| !self.name_to_idx.contains_key(*name))
                .cloned().unwrap_or_default();
            return Err(format!("Unknown condition variable '{}'", unknown));
        }

        let identified = if condition_names.is_empty() {
            self.identify(query_outcomes, &treatment_names)
        } else {
            self.identify_conditional(query_outcomes, &treatment_names, &condition_names)
        };
        if !identified.identifiable {
            return Err(format!(
                "Query is not identifiable ({}): {}",
                identified.method, identified.formula
            ));
        }
        let formula_ast = identified.formula_ast.ok_or_else(|| {
            "Identifiable query did not include a formula AST".to_string()
        })?;
        let distribution = formula_ast.evaluate_distribution(dataset, interventions, conditions)?;
        Ok(CausalEstimate {
            method: identified.method,
            formula: identified.formula,
            formula_ast,
            sample_size: dataset.rows.len(),
            distribution,
        })
    }

    /// Compute exact interventional causal expectation E[Y | do(X = val)]
    pub fn causal_effect(&self, treatment: &str, outcome: &str, val: f64) -> Result<f64, String> {
        let &x = self.name_to_idx.get(treatment)
            .ok_or_else(|| format!("Unknown treatment '{}'", treatment))?;
        let &y = self.name_to_idx.get(outcome)
            .ok_or_else(|| format!("Unknown outcome '{}'", outcome))?;

        let id = self.identify(&[outcome.to_string()], &[treatment.to_string()]);
        if !id.identifiable {
            return Err(format!("Causal effect of '{}' on '{}' is not identifiable: {}", treatment, outcome, id.formula));
        }

        let paths_effect = self.total_directed_effect(x, y);
        let baseline = self.noise_means[y];
        Ok(baseline + paths_effect * val)
    }

    fn total_directed_effect(&self, start: usize, target: usize) -> f64 {
        if start == target {
            return 1.0;
        }
        let mut total = 0.0;
        for next in self.children(start) {
            let weight = self.weights[start][next];
            total += weight * self.total_directed_effect(next, target);
        }
        total
    }

    /// Confounding Bias = E[Y | X=val] - E[Y | do(X=val)]
    pub fn confounding_bias(&self, treatment: &str, outcome: &str, val: f64) -> Result<f64, String> {
        let true_causal = self.causal_effect(treatment, outcome, val)?;
        let mut obs_map = HashMap::new();
        obs_map.insert(treatment.to_string(), val);
        let obs_res = self.observe(&obs_map)
            .ok_or_else(|| "Failed to compute observational inference".to_string())?;
        let (obs_mean, _) = obs_res.get(outcome)
            .ok_or_else(|| format!("Outcome '{}' not found in observational result", outcome))?;
        Ok(*obs_mean - true_causal)
    }
}

// ═══════════════════════════════════════════
//  Causal Discovery: PC Algorithm
// ═══════════════════════════════════════════

#[derive(Clone, Debug)]
pub struct PCResult {
    /// 1.0 indicates a directed edge from i -> j, 0.5 indicates an undirected edge, 0.0 is no edge.
    pub adjacency: Vec<Vec<f64>>,
    /// Confidence or strength of correlation for the edges.
    pub confidences: Vec<Vec<f64>>,
    pub names: Vec<String>,
}

/// Run PC Algorithm for Causal Discovery
pub fn discover(data: &[Vec<f64>], names: Vec<String>, alpha: f64) -> PCResult {
    let n_vars = names.len();
    let n_samples = data.len();

    // 1. Compute empirical correlation matrix
    let mut means = vec![0.0; n_vars];
    for row in data {
        for j in 0..n_vars {
            means[j] += row[j];
        }
    }
    for j in 0..n_vars {
        means[j] /= n_samples as f64;
    }

    let mut cov = vec![vec![0.0; n_vars]; n_vars];
    for row in data {
        for i in 0..n_vars {
            for j in 0..n_vars {
                cov[i][j] += (row[i] - means[i]) * (row[j] - means[j]);
            }
        }
    }
    for i in 0..n_vars {
        for j in 0..n_vars {
            cov[i][j] /= (n_samples - 1) as f64;
        }
    }

    // Undirected graph skeleton initialized to complete graph (except self-loops)
    let mut adj = vec![vec![true; n_vars]; n_vars];
    for i in 0..n_vars {
        adj[i][i] = false;
    }

    // Separating sets
    let mut sepsets = HashMap::<(usize, usize), Vec<usize>>::new();

    // Skeleton discovery
    let mut depth = 0;
    loop {
        let mut stable = true;
        for i in 0..n_vars {
            for j in 0..n_vars {
                if !adj[i][j] {
                    continue;
                }

                // Get neighbors of i excluding j
                let mut neighbors = Vec::new();
                for k in 0..n_vars {
                    if adj[i][k] && k != j {
                        neighbors.push(k);
                    }
                }

                if neighbors.len() < depth {
                    continue;
                }

                stable = false;

                let combinations = get_combinations(&neighbors, depth);
                for cond_set in combinations {
                    // Test conditional independence of i and j given cond_set
                    let p_val = test_conditional_independence(&cov, i, j, &cond_set, n_samples);
                    if p_val > alpha {
                        adj[i][j] = false;
                        adj[j][i] = false;
                        sepsets.insert((i, j), cond_set.clone());
                        sepsets.insert((j, i), cond_set.clone());
                        break;
                    }
                }
            }
        }

        if stable {
            break;
        }
        depth += 1;
        if depth >= n_vars {
            break;
        }
    }

    // Output representation: directed graph. Initialize as undirected (0.5 in both directions)
    // if adj is true.
    let mut res_adj = vec![vec![0.0; n_vars]; n_vars];
    for i in 0..n_vars {
        for j in 0..n_vars {
            if adj[i][j] {
                res_adj[i][j] = 0.5;
            }
        }
    }

    // 2. Orient V-structures (colliders): i - k - j where i and j are not adjacent
    for i in 0..n_vars {
        for j in 0..n_vars {
            if i >= j || adj[i][j] {
                continue;
            }
            // Look for common neighbors k
            for k in 0..n_vars {
                if adj[i][k] && adj[j][k] {
                    // Check separating set
                    let in_sepset = if let Some(set) = sepsets.get(&(i, j)) {
                        set.contains(&k)
                    } else {
                        false
                    };
                    if !in_sepset {
                        // Orient: i -> k <- j
                        res_adj[i][k] = 1.0;
                        res_adj[k][i] = 0.0;
                        res_adj[j][k] = 1.0;
                        res_adj[k][j] = 0.0;
                    }
                }
            }
        }
    }

    // 3. Meek's orientation rules
    loop {
        let mut changed = false;

        // Rule 1: Orient k - j into k -> j if there is i -> k and i, j are not adjacent
        for i in 0..n_vars {
            for k in 0..n_vars {
                if res_adj[i][k] == 1.0 {
                    for j in 0..n_vars {
                        if res_adj[k][j] == 0.5 && res_adj[j][k] == 0.5 && !adj[i][j] && i != j {
                            res_adj[k][j] = 1.0;
                            res_adj[j][k] = 0.0;
                            changed = true;
                        }
                    }
                }
            }
        }

        // Rule 2: Orient i - j into i -> j if there is a chain i -> k -> j
        for i in 0..n_vars {
            for k in 0..n_vars {
                if res_adj[i][k] == 1.0 {
                    for j in 0..n_vars {
                        if res_adj[k][j] == 1.0 && res_adj[i][j] == 0.5 && res_adj[j][i] == 0.5 {
                            res_adj[i][j] = 1.0;
                            res_adj[j][i] = 0.0;
                            changed = true;
                        }
                    }
                }
            }
        }

        // Rule 3: Orient i - j into i -> j if there are two chains i - k -> j and i - l -> j with k, l not adjacent
        for i in 0..n_vars {
            for j in 0..n_vars {
                if res_adj[i][j] == 0.5 && res_adj[j][i] == 0.5 {
                    for k in 0..n_vars {
                        for l in 0..n_vars {
                            if k != l && !adj[k][l] &&
                               res_adj[i][k] == 0.5 && res_adj[k][i] == 0.5 && res_adj[k][j] == 1.0 &&
                               res_adj[i][l] == 0.5 && res_adj[l][i] == 0.5 && res_adj[l][j] == 1.0
                            {
                                res_adj[i][j] = 1.0;
                                res_adj[j][i] = 0.0;
                                changed = true;
                            }
                        }
                    }
                }
            }
        }

        if apply_meek_rule4(&adj, &mut res_adj) {
            changed = true;
        }

        if !changed {
            break;
        }
    }

    // Compute edge confidences based on absolute partial correlation (1 - p_value)
    let mut confidences = vec![vec![0.0; n_vars]; n_vars];
    for i in 0..n_vars {
        for j in 0..n_vars {
            if adj[i][j] {
                // Retrieve the separating set if they were independent, else empty set
                let set = sepsets.get(&(i, j)).cloned().unwrap_or(vec![]);
                let p_val = test_conditional_independence(&cov, i, j, &set, n_samples);
                confidences[i][j] = (1.0 - p_val).max(0.0).min(1.0);
            }
        }
    }

    PCResult {
        adjacency: res_adj,
        confidences,
        names,
    }
}

/// Apply Meek Rule 4 to a partially directed graph.
/// If a - b, d is adjacent to a, a is adjacent to c, d -> c -> b,
/// and d and b are nonadjacent, orient a - b as a -> b.
fn apply_meek_rule4(skeleton: &[Vec<bool>], graph: &mut [Vec<f64>]) -> bool {
    let n = graph.len();
    let mut changed = false;
    for a in 0..n {
        for b in 0..n {
            if a == b || graph[a][b] != 0.5 || graph[b][a] != 0.5 {
                continue;
            }

            let mut rule_applies = false;
            'r4_pattern: for d in 0..n {
                if d == a || d == b || !skeleton[d][a] || skeleton[b][d] {
                    continue;
                }
                for c in 0..n {
                    if c == a || c == b || c == d || !skeleton[a][c] {
                        continue;
                    }
                    if graph[d][c] == 1.0
                        && graph[c][d] == 0.0
                        && graph[c][b] == 1.0
                        && graph[b][c] == 0.0
                    {
                        rule_applies = true;
                        break 'r4_pattern;
                    }
                }
            }

            if rule_applies {
                graph[a][b] = 1.0;
                graph[b][a] = 0.0;
                changed = true;
            }
        }
    }
    changed
}

/// Compute combinations of size `k` from `n` items
fn get_combinations(items: &[usize], k: usize) -> Vec<Vec<usize>> {
    let mut results = Vec::new();
    let mut combo = vec![0; k];
    combinations_recurse(items, k, 0, 0, &mut combo, &mut results);
    results
}

fn combinations_recurse(
    items: &[usize],
    k: usize,
    depth: usize,
    next: usize,
    combo: &mut [usize],
    results: &mut Vec<Vec<usize>>,
) {
    if depth == k {
        results.push(combo.to_vec());
        return;
    }
    for i in next..items.len() {
        combo[depth] = items[i];
        combinations_recurse(items, k, depth + 1, i + 1, combo, results);
    }
}

/// Test conditional independence using partial correlation & Fisher's z-transform
fn test_conditional_independence(
    cov: &[Vec<f64>],
    i: usize,
    j: usize,
    cond: &[usize],
    n_samples: usize,
) -> f64 {
    if cond.is_empty() {
        // Zero-order correlation
        let denom = (cov[i][i] * cov[j][j]).sqrt();
        if denom.abs() < 1e-12 {
            return 1.0; // Independent if variance is 0
        }
        let r = cov[i][j] / denom;
        return fisher_z_test(r, 0, n_samples);
    }

    let idx_a = vec![i, j];
    let idx_b = cond.to_vec();

    // Sigma_AA (2 x 2)
    let sig_aa = vec![
        vec![cov[idx_a[0]][idx_a[0]], cov[idx_a[0]][idx_a[1]]],
        vec![cov[idx_a[1]][idx_a[0]], cov[idx_a[1]][idx_a[1]]],
    ];

    // Sigma_BB (d x d)
    let d = idx_b.len();
    let mut sig_bb = vec![vec![0.0; d]; d];
    for r in 0..d {
        for c in 0..d {
            sig_bb[r][c] = cov[idx_b[r]][idx_b[c]];
        }
    }

    let sig_bb_inv = match invert_matrix(&sig_bb) {
        Some(inv) => inv,
        None => return 1.0, // Treat as independent if conditioning covariance is singular
    };

    // Sigma_AB (2 x d)
    let mut sig_ab = vec![vec![0.0; d]; 2];
    for r in 0..2 {
        for c in 0..d {
            sig_ab[r][c] = cov[idx_a[r]][idx_b[c]];
        }
    }

    // Sigma_BA (d x 2)
    let mut sig_ba = vec![vec![0.0; 2]; d];
    for r in 0..d {
        for c in 0..2 {
            sig_ba[r][c] = cov[idx_b[r]][idx_a[c]];
        }
    }

    // Sigma_A|B = Sigma_AA - Sigma_AB * Sigma_BB^-1 * Sigma_BA
    // sig_ab * sig_bb_inv (2 x d)
    let mut tmp = vec![vec![0.0; d]; 2];
    for r in 0..2 {
        for c in 0..d {
            for k in 0..d {
                tmp[r][c] += sig_ab[r][k] * sig_bb_inv[k][c];
            }
        }
    }

    // tmp * sig_ba (2 x 2)
    let mut deduct = vec![vec![0.0; 2]; 2];
    for r in 0..2 {
        for c in 0..2 {
            for k in 0..d {
                deduct[r][c] += tmp[r][k] * sig_ba[k][c];
            }
        }
    }

    let sig_a_cond_b = vec![
        vec![sig_aa[0][0] - deduct[0][0], sig_aa[0][1] - deduct[0][1]],
        vec![sig_aa[1][0] - deduct[1][0], sig_aa[1][1] - deduct[1][1]],
    ];

    let denom = (sig_a_cond_b[0][0] * sig_a_cond_b[1][1]).sqrt();
    if denom.abs() < 1e-12 {
        return 1.0;
    }
    let r_partial = sig_a_cond_b[0][1] / denom;
    fisher_z_test(r_partial, d, n_samples)
}

fn fisher_z_test(r: f64, cond_len: usize, n_samples: usize) -> f64 {
    // Clip r to avoid division by zero or log of negative
    let r_clipped = r.max(-0.999999).min(0.999999);
    let z = 0.5 * ((1.0 + r_clipped) / (1.0 - r_clipped)).ln();
    let df = (n_samples as f64 - cond_len as f64 - 3.0).max(1.0);
    let stat = z.abs() * df.sqrt();
    // 2 * (1 - Phi(stat))
    2.0 * (1.0 - std_normal_cdf(stat))
}

fn std_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
}

fn erf_approx(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    sign * y
}

// ═══════════════════════════════════════════
//  Matrix Helpers
// ═══════════════════════════════════════════

/// Check positive semidefiniteness using an LDLᵀ factorization that permits
/// zero pivots, as covariance matrices may be singular.
fn is_positive_semidefinite(matrix: &[Vec<f64>]) -> bool {
    let n = matrix.len();
    if matrix.iter().any(|row| row.len() != n || row.iter().any(|v| !v.is_finite())) {
        return false;
    }
    let scale = matrix.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()));
    let tolerance = scale * 1e-10;
    let mut lower = vec![vec![0.0; n]; n];
    let mut diagonal = vec![0.0; n];

    for i in 0..n {
        lower[i][i] = 1.0;
        let mut pivot = matrix[i][i];
        for k in 0..i {
            pivot -= lower[i][k] * lower[i][k] * diagonal[k];
        }
        if pivot < -tolerance {
            return false;
        }

        if pivot <= tolerance {
            diagonal[i] = 0.0;
            for j in (i + 1)..n {
                let mut residual = matrix[j][i];
                for k in 0..i {
                    residual -= lower[j][k] * lower[i][k] * diagonal[k];
                }
                if residual.abs() > tolerance {
                    return false;
                }
                lower[j][i] = 0.0;
            }
        } else {
            diagonal[i] = pivot;
            for j in (i + 1)..n {
                let mut residual = matrix[j][i];
                for k in 0..i {
                    residual -= lower[j][k] * lower[i][k] * diagonal[k];
                }
                lower[j][i] = residual / pivot;
            }
        }
    }

    true
}

fn invert_matrix(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = m.len();
    let mut a = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = m[i][j];
        }
        a[i][n + i] = 1.0;
    }
    for i in 0..n {
        // Find pivot
        let mut pivot_row = i;
        for r in i+1..n {
            if a[r][i].abs() > a[pivot_row][i].abs() {
                pivot_row = r;
            }
        }
        if a[pivot_row][i].abs() < 1e-12 {
            return None; // Singular
        }
        if pivot_row != i {
            a.swap(i, pivot_row);
        }
        let pivot = a[i][i];
        for j in i..2*n {
            a[i][j] /= pivot;
        }
        for r in 0..n {
            if r != i {
                let factor = a[r][i];
                for j in i..2*n {
                    a[r][j] -= factor * a[i][j];
                }
            }
        }
    }
    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = a[i][n + j];
        }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_variable_model() -> CausalModel {
        CausalModel::new(
            vec!["A".into(), "B".into()],
            vec![vec![0.0; 2]; 2],
            vec![1.0; 2],
            vec![0.0; 2],
        )
    }

    fn correlated_two_variable_model() -> CausalModel {
        let mut model = two_variable_model();
        model.add_bidirected_with_covariance("A", "B", 0.8).unwrap();
        model
    }

    #[test]
    fn bidirected_covariance_changes_conditional_observation() {
        let model = correlated_two_variable_model();
        let evidence = HashMap::from([("A".to_string(), 1.0)]);
        let result = model.observe(&evidence).unwrap();
        let (mean, std) = result["B"];

        assert!((mean - 0.8).abs() < 1e-9);
        assert!((std - 0.6).abs() < 1e-9);
    }

    #[test]
    fn counterfactual_abduction_uses_correlated_noise() {
        let model = correlated_two_variable_model();
        let evidence = HashMap::from([("A".to_string(), 1.0)]);
        let intervention = HashMap::from([("A".to_string(), 0.0)]);
        let result = model.counterfactual(&evidence, &intervention, &["B".to_string()]).unwrap();

        assert!((result["B"] - 0.8).abs() < 1e-9);
    }

    #[test]
    fn intervention_removes_shared_noise_with_target() {
        let model = correlated_two_variable_model();
        let intervention = HashMap::from([("A".to_string(), 1.0)]);
        let result = model.intervene(&intervention).unwrap();
        let (mean, std) = result["B"];

        assert!(mean.abs() < 1e-9);
        assert!((std - 1.0).abs() < 1e-9);
    }

    #[test]
    fn invalid_bidirected_covariance_is_rejected_without_mutation() {
        let mut model = two_variable_model();

        assert!(model.add_bidirected_with_covariance("A", "B", 1.2).is_err());
        assert!(model.bidirected.is_empty());
        assert!(model.noise_covariances.is_empty());
    }

    #[test]
    fn recursive_id_identifies_frontdoor_with_external_q_factor_context() {
        let mut weights = vec![vec![0.0; 4]; 4];
        weights[0][1] = 1.0; // X -> M
        weights[1][2] = 1.0; // M -> Y
        let mut model = CausalModel::new(
            vec!["X".into(), "M".into(), "Y".into(), "U".into()],
            weights,
            vec![1.0; 4],
            vec![0.0; 4],
        );
        model.add_bidirected("X", "Y");

        // U forces the general multi-treatment API path; ID's recursive Q[Y] must
        // retain M as a conditioning context while it marginalizes X.
        let result = model.identify(&["Y".into()], &["X".into(), "U".into()]);

        assert!(result.identifiable, "{}", result.formula);
        assert_eq!(result.method, "ID");
        assert!(result.formula.contains("Q"), "{}", result.formula);
        assert!(result.formula.contains("Y | M"), "{}", result.formula);
        assert!(result.formula.contains("Y | X, M"), "{}", result.formula);
    }

    #[test]
    fn recursive_id_identifies_canonical_napkin_graph() {
        let mut weights = vec![vec![0.0; 4]; 4];
        weights[0][1] = 1.0; // W -> Z
        weights[1][2] = 1.0; // Z -> X
        weights[2][3] = 1.0; // X -> Y
        let mut model = CausalModel::new(
            vec!["W".into(), "Z".into(), "X".into(), "Y".into()],
            weights,
            vec![1.0; 4],
            vec![0.0; 4],
        );
        model.add_bidirected("W", "X");
        model.add_bidirected("W", "Y");

        let result = model.identify(&["Y".into()], &["X".into()]);

        assert!(result.identifiable, "{}", result.formula);
        assert_eq!(result.method, "ID");
        let formula_ast = result.formula_ast.as_ref().expect("identifiable result should carry a formula AST");
        assert_eq!(formula_ast.render(), result.formula);
        assert_eq!(formula_ast.outcomes, vec!["Y"]);
        assert!(!formula_ast.definitions.is_empty());
        assert!(matches!(formula_ast.definitions[0].expression, IdentificationExpr::Product(_)));
        assert!(result.formula.contains("sum_{Z}"), "{}", result.formula);
        assert!(result.formula.contains("P(Z)"), "{}", result.formula);
    }

    #[test]
    fn idc_exchanges_a_condition_when_the_mutilated_graph_separates_it() {
        let mut weights = vec![vec![0.0; 3]; 3];
        weights[0][1] = 1.0; // X -> Z
        weights[1][2] = 1.0; // Z -> Y
        let model = CausalModel::new(
            vec!["X".into(), "Z".into(), "Y".into()],
            weights,
            vec![1.0; 3],
            vec![0.0; 3],
        );

        let result = model.identify_conditional(&["Y".into()], &["X".into()], &["Z".into()]);

        assert!(result.identifiable, "{}", result.formula);
        assert_eq!(result.method, "IDC");
        assert!(result.formula.contains("P(Y | do(X), Z)"), "{}", result.formula);
        let formula = result.formula_ast.expect("IDC should return a structured formula");
        assert_eq!(formula.conditions, vec!["Z"]);
        assert!(!matches!(formula.expression, IdentificationExpr::Quotient { .. }));
        assert_eq!(formula.render(), result.formula);
    }

    #[test]
    fn idc_builds_a_joint_id_over_marginal_ratio_when_exchange_fails() {
        let mut weights = vec![vec![0.0; 3]; 3];
        weights[0][1] = 1.0; // X -> Y
        let mut model = CausalModel::new(
            vec!["X".into(), "Y".into(), "Z".into()],
            weights,
            vec![1.0; 3],
            vec![0.0; 3],
        );
        model.add_bidirected("Y", "Z");

        let result = model.identify_conditional(&["Y".into()], &["X".into()], &["Z".into()]);

        assert!(result.identifiable, "{}", result.formula);
        assert_eq!(result.method, "IDC");
        assert!(result.formula.contains("P(Y | do(X), Z)"), "{}", result.formula);
        let formula = result.formula_ast.expect("IDC should return a structured formula");
        assert_eq!(formula.conditions, vec!["Z"]);
        let IdentificationExpr::Quotient { denominator, .. } = &formula.expression else {
            panic!("IDC should normalize the joint distribution when exchange fails");
        };
        assert!(matches!(denominator.as_ref(), IdentificationExpr::Sum { variables, .. } if variables == &["Y"]));
        assert_eq!(formula.render(), result.formula);
    }

    #[test]
    fn categorical_estimator_applies_backdoor_formula_to_observed_rows() {
        let mut weights = vec![vec![0.0; 3]; 3];
        weights[0][1] = 1.0; // Z -> X
        weights[0][2] = 1.0; // Z -> Y
        let model = CausalModel::new(
            vec!["Z".into(), "X".into(), "Y".into()],
            weights,
            vec![1.0; 3],
            vec![0.0; 3],
        );
        let mut rows = Vec::new();
        rows.extend((0..3).map(|_| vec!["0".into(), "0".into(), "0".into()]));
        rows.push(vec!["0".into(), "1".into(), "1".into()]);
        rows.push(vec!["1".into(), "0".into(), "0".into()]);
        rows.extend((0..3).map(|_| vec!["1".into(), "1".into(), "0".into()]));
        let dataset = CausalDataset::new(vec!["Z".into(), "X".into(), "Y".into()], rows).unwrap();

        let estimate = model.estimate_distribution(
            &dataset,
            &["Y".into()],
            &HashMap::from([("X".into(), "1".into())]),
            &HashMap::new(),
        ).unwrap();

        assert_eq!(estimate.method, "Backdoor");
        assert_eq!(estimate.sample_size, 8);
        assert!((estimate.distribution.iter().map(|entry| entry.probability).sum::<f64>() - 1.0).abs() < 1e-12);
        let treated_probability = estimate.distribution.iter()
            .find(|entry| entry.values == ["1"])
            .unwrap().probability;
        assert!((treated_probability - 0.5).abs() < 1e-12);
    }

    #[test]
    fn categorical_dataset_rejects_ragged_rows_and_estimator_checks_support() {
        assert!(CausalDataset::new(
            vec!["X".into(), "Y".into()],
            vec![vec!["0".into()]],
        ).unwrap_err().contains("expected 2"));

        let mut weights = vec![vec![0.0; 2]; 2];
        weights[0][1] = 1.0;
        let model = CausalModel::new(
            vec!["X".into(), "Y".into()],
            weights,
            vec![1.0; 2],
            vec![0.0; 2],
        );
        let dataset = CausalDataset::new(
            vec!["X".into(), "Y".into()],
            vec![vec!["0".into(), "0".into()]],
        ).unwrap();
        let error = model.estimate_distribution(
            &dataset,
            &["Y".into()],
            &HashMap::from([("X".into(), "1".into())]),
            &HashMap::new(),
        ).unwrap_err();
        assert!(error.contains("no observations for category '1'"), "{}", error);
    }

    #[test]
    fn categorical_estimator_evaluates_idc_quotient_formula() {
        let mut weights = vec![vec![0.0; 3]; 3];
        weights[0][1] = 1.0; // X -> Y
        let mut model = CausalModel::new(
            vec!["X".into(), "Y".into(), "Z".into()],
            weights,
            vec![1.0; 3],
            vec![0.0; 3],
        );
        model.add_bidirected("Y", "Z");
        let mut rows = Vec::new();
        for x in ["0", "1"] {
            for y in ["0", "1"] {
                for z in ["0", "1"] {
                    rows.push(vec![x.into(), y.into(), z.into()]);
                }
            }
        }
        let dataset = CausalDataset::new(vec!["X".into(), "Y".into(), "Z".into()], rows).unwrap();

        let estimate = model.estimate_distribution(
            &dataset,
            &["Y".into()],
            &HashMap::from([("X".into(), "1".into())]),
            &HashMap::from([("Z".into(), "1".into())]),
        ).unwrap();

        assert_eq!(estimate.method, "IDC");
        assert!(matches!(estimate.formula_ast.expression, IdentificationExpr::Quotient { .. }));
        for entry in estimate.distribution {
            assert!((entry.probability - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn categorical_estimator_handles_frontdoor_treatment_summation() {
        let mut weights = vec![vec![0.0; 3]; 3];
        weights[0][1] = 1.0; // X -> M
        weights[1][2] = 1.0; // M -> Y
        let mut model = CausalModel::new(
            vec!["X".into(), "M".into(), "Y".into()],
            weights,
            vec![1.0; 3],
            vec![0.0; 3],
        );
        model.add_bidirected("X", "Y");
        let mut rows = Vec::new();
        for x in ["0", "1"] {
            for m in ["0", "1"] {
                for y in ["0", "1"] {
                    rows.push(vec![x.into(), m.into(), y.into()]);
                }
            }
        }
        let dataset = CausalDataset::new(vec!["X".into(), "M".into(), "Y".into()], rows).unwrap();

        let estimate = model.estimate_distribution(
            &dataset,
            &["Y".into()],
            &HashMap::from([("X".into(), "1".into())]),
            &HashMap::new(),
        ).unwrap();

        assert_eq!(estimate.method, "Frontdoor");
        for entry in estimate.distribution {
            assert!((entry.probability - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn categorical_estimator_evaluates_recursive_q_factor_definitions() {
        let mut weights = vec![vec![0.0; 4]; 4];
        weights[0][1] = 1.0; // W -> Z
        weights[1][2] = 1.0; // Z -> X
        weights[2][3] = 1.0; // X -> Y
        let mut model = CausalModel::new(
            vec!["W".into(), "Z".into(), "X".into(), "Y".into()],
            weights,
            vec![1.0; 4],
            vec![0.0; 4],
        );
        model.add_bidirected("W", "X");
        model.add_bidirected("W", "Y");
        let mut rows = Vec::new();
        for w in ["0", "1"] {
            for z in ["0", "1"] {
                for x in ["0", "1"] {
                    for y in ["0", "1"] {
                        rows.push(vec![w.into(), z.into(), x.into(), y.into()]);
                    }
                }
            }
        }
        let dataset = CausalDataset::new(vec!["W".into(), "Z".into(), "X".into(), "Y".into()], rows).unwrap();

        let estimate = model.estimate_distribution(
            &dataset,
            &["Y".into()],
            &HashMap::from([("X".into(), "1".into())]),
            &HashMap::new(),
        ).unwrap();

        assert_eq!(estimate.method, "ID");
        assert!(!estimate.formula_ast.definitions.is_empty());
        assert!(estimate.formula.contains("sum_{Z}"), "{}", estimate.formula);
        assert!(estimate.formula.contains("P(Z)"), "{}", estimate.formula);
        for entry in estimate.distribution {
            assert!((entry.probability - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn recursive_id_returns_outer_and_inner_hedge_for_bow_graph() {
        let mut weights = vec![vec![0.0; 2]; 2];
        weights[0][1] = 1.0; // X -> Y
        let mut model = CausalModel::new(
            vec!["X".into(), "Y".into()],
            weights,
            vec![1.0; 2],
            vec![0.0; 2],
        );
        model.add_bidirected("X", "Y");

        let result = model.identify(&["Y".into()], &["X".into()]);

        assert!(!result.identifiable);
        assert_eq!(result.method, "ID-Hedge");
        assert_eq!(result.hedge_outer, Some(vec!["X".into(), "Y".into()]));
        assert_eq!(result.hedge_inner, Some(vec!["Y".into()]));
    }

    #[test]
    fn recursive_id_handles_joint_multi_outcome_multi_treatment_queries() {
        let mut weights = vec![vec![0.0; 4]; 4];
        weights[0][1] = 1.0; // X -> Y
        weights[2][3] = 1.0; // A -> B
        let model = CausalModel::new(
            vec!["X".into(), "Y".into(), "A".into(), "B".into()],
            weights,
            vec![1.0; 4],
            vec![0.0; 4],
        );

        let result = model.identify(&["Y".into(), "B".into()], &["X".into(), "A".into()]);

        assert!(result.identifiable, "{}", result.formula);
        assert_eq!(result.method, "ID");
        assert!(result.formula.contains("P(Y, B | do(X, A))"), "{}", result.formula);
        assert!(result.formula.contains(" * "), "{}", result.formula);
    }

    #[test]
    fn meek_rule4_orients_only_when_its_pattern_is_present() {
        let mut skeleton = vec![vec![false; 4]; 4];
        for (u, v) in [(0, 1), (3, 0), (0, 2), (3, 2), (2, 1)] {
            skeleton[u][v] = true;
            skeleton[v][u] = true;
        }
        let mut graph = vec![vec![0.0; 4]; 4];
        graph[0][1] = 0.5;
        graph[1][0] = 0.5;
        graph[3][0] = 0.5;
        graph[0][3] = 0.5;
        graph[0][2] = 0.5;
        graph[2][0] = 0.5;
        graph[3][2] = 1.0;
        graph[2][3] = 0.0;
        graph[2][1] = 1.0;
        graph[1][2] = 0.0;

        assert!(apply_meek_rule4(&skeleton, &mut graph));
        assert_eq!(graph[0][1], 1.0);
        assert_eq!(graph[1][0], 0.0);

        let mut blocked_skeleton = skeleton.clone();
        blocked_skeleton[1][3] = true;
        blocked_skeleton[3][1] = true;
        let mut blocked_graph = graph.clone();
        blocked_graph[0][1] = 0.5;
        blocked_graph[1][0] = 0.5;
        assert!(!apply_meek_rule4(&blocked_skeleton, &mut blocked_graph));
        assert_eq!(blocked_graph[0][1], 0.5);
        assert_eq!(blocked_graph[1][0], 0.5);
    }
}
