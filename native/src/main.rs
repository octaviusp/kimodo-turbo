//! Deterministic humanoid fitting and surface-aware skin weights for standalone GLB files.
mod anatomy;
mod appearance;
mod capsules;
mod cloth;
mod diffusion;
mod garment;
mod glb;
mod mesh;
mod weights;

use glam::Vec3;
use serde_json::json;
use std::{fs, path::PathBuf, time::Instant};

fn axis(value: &str) -> Result<Vec3, String> {
    match value.to_ascii_lowercase().as_str() {
        "x" | "+x" => Ok(Vec3::X),
        "-x" => Ok(-Vec3::X),
        "y" | "+y" => Ok(Vec3::Y),
        "-y" => Ok(-Vec3::Y),
        "z" | "+z" => Ok(Vec3::Z),
        "-z" => Ok(-Vec3::Z),
        _ => Err(format!(
            "Invalid axis {value:?}; use x, y, z or their negative forms"
        )),
    }
}
fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args[0] == "--help" || args[0] == "-h" {
        println!(
            "kimodo-autorig INPUT.glb OUTPUT.glb [--report FILE.json] [--force] [--up AXIS] [--forward AXIS]\n\nAxes: x, y, z, -x, -y, -z. Up and forward must be perpendicular.\nPreserves authored rigs unless --force is supplied. Automatic rigs are rebuilt.\nThe input is never overwritten. Output is deterministic for identical inputs, options and build."
        );
        return Ok(());
    }
    if args.len() < 2 {
        return Err("Specify an input and output GLB; use --help for usage".into());
    }
    let input = PathBuf::from(&args[0]);
    let output = PathBuf::from(&args[1]);
    if input.canonicalize().ok() == output.canonicalize().ok() && input.exists() && output.exists()
    {
        return Err("Input and output must be different files".into());
    }
    let (mut report_path, mut up, mut forward, mut force) = (None, None, None, false);
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--force" => {
                force = true;
                i += 1;
            }
            "--report" | "--up" | "--forward" => {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| format!("{} requires a value", args[i]))?;
                match args[i].as_str() {
                    "--report" => report_path = Some(PathBuf::from(value)),
                    "--up" => up = Some(axis(value)?),
                    _ => forward = Some(axis(value)?),
                }
                i += 2;
            }
            _ => return Err(format!("Unknown option {}", args[i])),
        }
    }
    if let (Some(up), Some(forward)) = (up, forward)
        && up.dot(forward).abs() > 0.01
    {
        return Err("Up and forward must be perpendicular".into());
    }
    if let Some(path) = &report_path
        && (path == &input
            || path == &output
            || (path.exists()
                && (path.canonicalize().ok() == input.canonicalize().ok()
                    || path.canonicalize().ok() == output.canonicalize().ok())))
    {
        return Err("The report must be a separate JSON file".into());
    }
    let start = Instant::now();
    let mut model = glb::Glb::read(&input)?;
    let report = if model.has_author_skin() && !force {
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let temporary = output.with_extension("glb.tmp");
        fs::write(&temporary, &model.original).map_err(|e| e.to_string())?;
        fs::rename(temporary, &output).map_err(|e| e.to_string())?;
        json!({"version":1,"mode":"preserved-author-skin","seconds":start.elapsed().as_secs_f64(),"notes":["Existing authored weights were preserved; geometry fitting was not evaluated."]})
    } else {
        let mut geometry = model.geometry()?;
        let surface = mesh::Surface::new(&geometry.points, &geometry.triangles);
        let fit_start = Instant::now();
        let skeleton = anatomy::fit(&surface.points, &surface.triangles, up, forward)?;
        let fit_seconds = fit_start.elapsed().as_secs_f64();
        let weight_start = Instant::now();
        let mut skin = weights::solve(&surface, &skeleton)?;
        if skeleton.covers_legs {
            let mut roles =
                cloth::lower_garment_faces(&geometry.points, &geometry.triangles, &skeleton);
            let appearance_report =
                appearance::refine_roles(&model, &geometry, &skeleton, &mut roles)?;
            garment::separate(&mut geometry, &skeleton, &mut skin, &roles)?;
            skin.report["garment_appearance"] = appearance_report;
        }
        let weight_seconds = weight_start.elapsed().as_secs_f64();
        model.bind(&geometry, &skeleton, &skin)?;
        model.write(&output)?;
        json!({"version":1,"mode":"generated","seconds":start.elapsed().as_secs_f64(),"fit_seconds":fit_seconds,"weight_seconds":weight_seconds,"vertices":geometry.points.len(),"triangles":geometry.triangles.len(),"height":skeleton.height,"up":skeleton.up.to_array(),"forward":skeleton.forward.to_array(),"covered_legs":skeleton.covers_legs,"joints":skeleton.names.iter().zip(skeleton.joints.iter()).map(|(n,p)|json!({"name":n,"position":p.to_array()})).collect::<Vec<_>>(),"notes":skeleton.notes,"weights":skin.report})
    };
    if let Some(path) = report_path {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).map_err(|e| e.to_string())?;
    }
    println!("{}", serde_json::to_string(&report).unwrap());
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("kimodo-autorig: {error}");
        std::process::exit(1);
    }
}
