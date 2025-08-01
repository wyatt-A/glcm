use std::collections::HashSet;
use std::env::current_dir;
use std::fs;
use std::fs::{File, OpenOptions};
use std::io::{Error, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::JoinHandle;
use array_lib::ArrayDim;
//use array_lib::io_nifti::{read_nifti, write_nifti_with_header};
use array_lib::io_nrrd::{read_nrrd_to_array, write_nrrd_from_array, Encoding};
use clap::Parser;
use glcm::ui::{GLCMFeature, GLCMFeatureIter, RadMapOpts, RadMapOptsSer};
use eframe;
use eframe::{egui, CreationContext};
use eframe::egui::{IconData, ProgressBar, TextEdit};
use strum::IntoEnumIterator;
use glcm::{change_dims, discretize_bin_count, generate_angles, n_angles};
use glcm::glcm::map_glcm;
use glcm::icon::ICON_DATA;

const DEFAULT_CONFIG_NAME:&str = "radmap_cfg.toml";

#[derive(Parser)]
struct Args {

    /// input volume
    #[clap(required_unless_present = "gen_config")]
    #[clap(required_unless_present = "gui")]
    input:Option<PathBuf>,

    /// output directory
    #[clap(required_unless_present = "gen_config")]
    #[clap(required_unless_present = "gui")]
    output_dir:Option<PathBuf>,

    /// generate a template config file for future runs
    #[clap(long)]
    gen_config:bool,

    /// launch the interactive gui
    #[clap(long)]
    gui:bool

}

fn main() {

    let args = Args::parse();

    // generate a new config file
    if args.gen_config {

        let opts = RadMapOpts::default();
        let ser:RadMapOptsSer = opts.into();
        let t_string = toml::to_string_pretty(&ser).expect("failed to serialize options");

        let f = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open("rad_map_cfg.toml");

        match f {
            Ok(mut file) => {
                file.write_all(t_string.as_bytes()).expect("failed to write to open file");
                println!("Successfully created rad_map_cfg.toml");
            }
            Err(e) => {
                if e.kind() == ErrorKind::AlreadyExists {
                    eprintln!("rad_map_cfg.toml already exists!");
                } else {
                    eprintln!("Failed to create file: {}", e);
                }
            }
        }
        return
    }

    if args.gui {

        let icon = get_icon().unwrap();
        let native_options = eframe::NativeOptions {
            //renderer: eframe::Renderer::Wgpu,
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1024.0, 768.0])
                .with_title("RadMap")
                .with_icon(icon),
            ..Default::default()
        };

        eframe::run_native(
            "RadMap",
            native_options,
            Box::new(|_cc| Ok(Box::new(MyApp::default()))),
        ).unwrap();
    }



}

struct MyApp {
    opts:Option<RadMapOpts>,
    loaded_config:Option<PathBuf>,
    kernel_rad_buffer: Option<String>,
    n_bins_buffer: Option<String>,


    valid_vol_path: Option<PathBuf>,
    valid_mask_path: Option<PathBuf>,
    valid_output_dir: Option<PathBuf>,

    vol_file_buffer: Option<String>,
    mask_file_buffer: Option<String>,
    output_dir_buffer: Option<String>,

    progress: Arc<AtomicUsize>,

    vox_to_process:Option<usize>,
    running: bool,
    process_handle: Option<JoinHandle<()>>,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            opts:None,
            loaded_config: None,
            kernel_rad_buffer: None,
            n_bins_buffer: None,
            vol_file_buffer: None,
            valid_vol_path: None,
            mask_file_buffer: None,
            valid_mask_path: None,
            progress: Arc::new(AtomicUsize::new(0)),
            vox_to_process: None,
            running: false,
            process_handle: None,
            valid_output_dir: None,
            output_dir_buffer: None,
        }
    }
}

impl eframe::App for MyApp {

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {

        let opts = self.opts.get_or_insert_with(|| {
            find_first_valid_toml(&current_dir().unwrap()).map(|(x,p)|{
                self.loaded_config = Some(p);
                x.into()
            })
                .unwrap_or_else(RadMapOpts::default)
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("GLCM RadMapper 9000");

            match &self.loaded_config {
                Some(p) => {
                    ui.heading(format!("{}",p.display()));
                }
                None => {
                    ui.heading("no config file found");
                }
            }

            ui.columns(2, |columns| {
                columns[0].vertical(|ui| {
                    ui.label("Feature Selection:");
                    feature_selector(opts,ui);
                });

                columns[1].vertical(|ui| {
                    ui.label("Parameter Selection:");
                    ui.horizontal(|ui| {
                        ui.label("Kernel Radius");
                        let krb = self.kernel_rad_buffer.get_or_insert(opts.kernel_radius.to_string());
                        let editor = TextEdit::singleline(krb)
                            .desired_width(40.0); // set width in logical pixels
                        let h = ui.add(editor);
                        if !h.has_focus() {
                            if let Ok(parsed) = krb.parse() {
                                let valid = if parsed == 0 {
                                    1
                                }else {
                                    parsed
                                };
                                opts.kernel_radius = valid;
                            }
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Number of Bins (GLCM size)");
                        let nbb = self.n_bins_buffer.get_or_insert(opts.n_bins.to_string());
                        let editor = TextEdit::singleline(nbb)
                            .desired_width(40.0); // set width in logical pixels
                        let h = ui.add(editor);
                        if !h.has_focus() {
                            if let Ok(parsed) = nbb.parse() {
                                let valid = if parsed < 1 {
                                    2
                                }else {
                                    parsed
                                };
                                opts.n_bins = valid;
                            }
                        }
                    });
                    ui.label("");
                    ui.label("Current Parameters:");
                    ui.horizontal(|ui| {
                        ui.label("Number of features:");
                        ui.label(opts.features.len().to_string());
                    });
                    ui.horizontal(|ui| {
                        ui.label("Kernel Radius:");
                        ui.label(opts.kernel_radius.to_string());
                    });
                    ui.horizontal(|ui| {
                        ui.label("Number of Bins (GLCM size):");
                        ui.label(opts.n_bins.to_string());
                    });

                    ui.label("");
                    ui.horizontal(|ui| {
                        ui.label("Volume: ");
                        let b = self.vol_file_buffer.get_or_insert(String::new());
                        let editor = TextEdit::singleline(b)
                            .desired_width(200.0); // set width in logical pixels
                        let h = ui.add(editor);
                        if !h.has_focus() {
                            if Path::new(b).exists() {
                                self.valid_vol_path = Some(PathBuf::from(b.as_str()));
                            }
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Mask: ");
                        let b = self.mask_file_buffer.get_or_insert(String::new());
                        let editor = TextEdit::singleline(b)
                            .desired_width(200.0); // set width in logical pixels
                        let h = ui.add(editor);
                        if !h.has_focus() {
                            if Path::new(b).exists() {
                                self.valid_mask_path = Some(PathBuf::from(b.as_str()));
                            }
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Output Directory: ");
                        let b = self.output_dir_buffer.get_or_insert(String::new());
                        let editor = TextEdit::singleline(b)
                            .desired_width(200.0); // set width in logical pixels
                        let h = ui.add(editor);
                        if !h.has_focus() {
                            if Path::new(b).exists() {
                                self.valid_output_dir = Some(PathBuf::from(b.as_str()));
                            }
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Vol path:");
                        if let Some(valid_vol_path) = &self.valid_vol_path {
                            ui.label(valid_vol_path.display().to_string());
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Mask path:");
                        if let Some(valid_mask_path) = &self.valid_mask_path {
                            ui.label(valid_mask_path.display().to_string());
                        }
                    });


                    if self.valid_vol_path.is_some() && self.valid_output_dir.is_some() {
                        if ui.button("GO!").clicked() {

                            self.progress = Arc::new(AtomicUsize::new(0));

                            let img_path = self.valid_vol_path.clone().unwrap();

                            let opts = opts.clone();
                            let prog_handle = self.progress.clone();

                            let h = std::thread::spawn(move || {
                                //read_nifti::<f64>(img_path)
                                read_nrrd_to_array(img_path)
                            });

                            let mask = if let Some(mask_path) = self.valid_mask_path.clone() {
                                let hm = std::thread::spawn(move || {
                                    read_nrrd_to_array::<f64>(mask_path)
                                });
                                let (msk,..) = hm.join().unwrap();
                                Some(msk)
                            }else {
                                None
                            };

                            let (img,img_dims,header) = h.join().unwrap();

                            self.vox_to_process = Some(img.len());

                            let out_dir = self.valid_output_dir.clone().unwrap();
                            let out_base = self.valid_vol_path.clone().unwrap();
                            let out_base = out_base.file_stem().unwrap().to_string_lossy().to_string();

                            let h = std::thread::spawn(move || {
                                let (out,dims) = run_map(opts.clone(), img, mask, img_dims, prog_handle);
                                let vol_stride:usize = img_dims.numel();
                                for (f,alias) in opts.features_aliases() {
                                    let i = f as usize;
                                    let vol = &out[i*vol_stride..(i+1) * vol_stride];
                                    let path = out_dir.join(format!("{}{}{}",out_base,opts.separator(),alias));
                                    write_nrrd_from_array(path,vol,img_dims,Some(&header),false,Encoding::raw);
                                    //write_nifti_with_header(path,vol,img_dims,&header);
                                }

                            });
                            self.running = true;
                            self.process_handle = Some(h);
                        }
                    }

                    if self.running {
                        let state = self.progress.load(Ordering::Relaxed);
                        let progress = state as f32 / self.vox_to_process.unwrap() as f32;
                        ui.add(ProgressBar::new(progress).show_percentage());
                        if self.process_handle.as_ref().unwrap().is_finished() {
                            ui.label("mapping succeeded");
                        }
                    }


                });
            });

            if ui.button("Save").clicked() {
                let cd = current_dir().unwrap();
                match opts.to_file(cd.join(DEFAULT_CONFIG_NAME)) {
                    Ok(_) => {
                        self.loaded_config = Some(cd.join(DEFAULT_CONFIG_NAME));
                    }
                    Err(e) => {
                        println!("Failed to save {DEFAULT_CONFIG_NAME}: {:?}", e);
                    }
                }
            }

            ctx.request_repaint();

        });
    }
}


/// Recursively search for the first `.toml` file starting from `start_dir`.
fn find_first_valid_toml(start_dir: &Path) -> Option<(RadMapOptsSer,PathBuf)> {
    for entry in fs::read_dir(start_dir).ok()? {
        let entry = entry.ok()?;
        let path = entry.path();

        if path.is_file() && path.extension().map_or(false, |ext| ext == "toml") {
            if let Ok(data) = fs::read_to_string(&path) {
                if let Ok(parsed) = toml::from_str(&data) {
                    return Some((parsed,path.to_path_buf()));
                }
            }
        } else if path.is_dir() {
            if let Some(found) = find_first_valid_toml(&path) {
                return Some(found);
            }
        }
    }
    None
}


fn feature_selector(opts:&mut RadMapOpts,ui: &mut egui::Ui) {
    for feature in GLCMFeature::iter() {
        let mut is_selected = opts.features.contains_key(&feature);
        if ui.checkbox(&mut is_selected, feature.to_string().replace("_"," ")).changed() {
            if is_selected {
                opts.features.insert(feature.clone(),feature.to_string().to_lowercase());
            } else {
                opts.features.remove(&feature);
            }
        }
    }
}

fn run_map(opts:RadMapOpts,image:Vec<f64>,mask:Option<Vec<f64>>,dims:ArrayDim,progress:Arc<AtomicUsize>) -> (Vec<f64>,ArrayDim) {

    let vol_dims = &dims.shape()[0..3];

    let mut bins = vec![0;dims.numel()];

    let mask = if let Some(mask) = mask {
        mask
    }else {
        vec![1.; dims.numel()]
    };

    let mask:Vec<u16> = mask.into_iter().map(|x| if x > 0. {1u16} else {0u16}).collect();

    discretize_bin_count(opts.n_bins,&image,&mask,&mut bins);

    let mut angles = vec![[0,0,0];n_angles(opts.kernel_radius)];
    generate_angles(&mut angles,opts.kernel_radius);

    let odims = ArrayDim::from_shape(&[24,vol_dims[0],vol_dims[1],vol_dims[2]]);

    let mut out = vec![0f64;odims.numel()];

    map_glcm(&opts.features(),vol_dims,&bins,&mut out,&angles,opts.n_bins,opts.kernel_radius,&[],progress);

    let mut features = odims.alloc(0f64);
    change_dims(vol_dims,24,&out,&mut features);

    let fdims =  ArrayDim::from_shape(&[vol_dims[0],vol_dims[1],vol_dims[2],24]);

    (features,fdims)

}

fn get_icon() -> Option<IconData> {
    let rgba = ICON_DATA.to_vec();
    Some(IconData {
        rgba,
        width:1024,
        height:1024,
    })
}