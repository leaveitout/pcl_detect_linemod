#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/point_cloud_image_extractors.h>
#include <pcl/recognition/linemod.h>
#include <pcl/recognition/color_gradient_modality.h>
#include <pcl/recognition/surface_normal_modality.h>
#include <pcl/visualization/image_viewer.h>

#include <boost/range/irange.hpp>
#include <boost/range/counting_range.hpp>
#include <boost/smart_ptr/make_shared.hpp>


// Typedefs
using PointType = pcl::PointXYZRGBA;
using Cloud = pcl::PointCloud <PointType>;


namespace fs = boost::filesystem;


// User defined literals
// @formatter:off
constexpr size_t operator "" _sz (unsigned long long size) { return size_t{size}; }
constexpr double operator "" _deg (long double deg) { return deg * M_PI / 180.0; }
constexpr double operator "" _deg (unsigned long long deg) { return deg * M_PI / 180.0; }
constexpr double operator "" _cm (long double cm) { return cm / 100.0; }
constexpr double operator "" _cm (unsigned long long cm) { return cm / 100.0; }
constexpr double operator "" _mm (long double mm) { return mm / 1000.0; }
constexpr double operator "" _mm (unsigned long long mm) { return mm / 1000.0; }
// @formatter:on


// Constants
constexpr auto MIN_VALID_ARGS = 3U;
constexpr auto MAX_VALID_ARGS = 3U;
//constexpr auto NUM_PCD_FILES_EXPECTED = 2U;
//constexpr auto NUM_PCD_DIRS_EXPECTED = 2U;
//constexpr auto INPUT_DIR_ARG_POS = 1U;
//constexpr auto OUTPUT_DIR_ARG_POS = 2U;
//constexpr auto DEFAULT_MIN_CLUSTER_SIZE = 100U;
//constexpr auto DEFAULT_MAX_CLUSTER_SIZE = 25000U;
//constexpr auto DEFAULT_TOLERANCE = 2_cm;
//constexpr auto DEFAULT_MAX_NUM_CLUSTERS = 20U;

template <typename T>
constexpr auto izrange (T upper)
-> decltype (boost::irange (static_cast<T> (0), upper)) {
  return boost::irange (static_cast <T> (0), upper);
}


auto printHelp (int argc, char ** argv)
-> void {
  using pcl::console::print_error;
  using pcl::console::print_info;

  // TODO: Update this help
  print_error ("Syntax is: %s (<path-to-pcd-file> <path-to-lmt-file>) <options> | "
                   "-h | --help\n", argv[0]);
  print_info ("%s -h | --help : shows this help\n", argv[0]);
  //  print_info ("-min X : use a minimum of X points per cluster (default: 100)\n");
  //  print_info ("-max X : use a maximum of X points per cluster (default: 25000)\n");
  //  print_info ("-tol X : the spatial distance (in meters) between clusters (default: 0.002.\n");
}


auto expandTilde (std::string path_string) -> fs::path {
  if (path_string.at (0) == '~')
    path_string.replace (0, 1, getenv ("HOME"));
  return fs::path{path_string};
}


auto checkValidFile (fs::path const & filepath)
-> bool {
  // Check that the file is valid
  return fs::exists (filepath) && fs::is_regular_file (filepath);
}


auto detectTemplates (Cloud::ConstPtr const & cloud, pcl::LINEMOD & linemod)
-> std::vector <pcl::LINEMODDetection> {
  auto color_grad_mod = pcl::ColorGradientModality <PointType> {};
  color_grad_mod.setInputCloud (cloud);
  color_grad_mod.processInputData ();

  auto surface_grad_mod = pcl::SurfaceNormalModality <PointType> {};
  surface_grad_mod.setInputCloud (cloud);
  surface_grad_mod.processInputData ();

  auto modalities = std::vector <pcl::QuantizableModality *> (2);
  modalities [0] = & color_grad_mod;
  modalities [1] = & surface_grad_mod;

  auto detections = std::vector <pcl::LINEMODDetection> {};
  linemod.matchTemplates (modalities, detections);

  return detections;
}

auto outputTemplateMatch (pcl::LINEMODDetection const & detection) -> void {
  std::cout << "x (" << detection.x << ") " <<
      "y (" << detection.y << ") " <<
      "id (" << detection.template_id << ") " <<
      "scale (" << detection.scale << ") " <<
      "score (" << detection.score << ")." << std::endl;
}

auto outputTemplateMatches (std::vector <pcl::LINEMODDetection> detections) -> void {
  auto index = 0UL;
  for (auto const & d : detections) {
    std::cout << "Detection " << index << " ";
    outputTemplateMatch (d);
    ++index;
  }
}


auto getClosestMatchedTemplate (std::vector <pcl::LINEMODDetection> detections)
-> pcl::LINEMODDetection {
  auto score_compare = [](pcl::LINEMODDetection const & a, pcl::LINEMODDetection const & b) {
    return a.score < b.score;
  };
  return *(std::max_element (detections.begin (), detections.end (), score_compare));
}


auto main (int argc, char * argv[])
-> int {
  pcl::console::print_highlight ("Tool to extract the largest cluster found in a point cloud.\n");

  auto help_flag_1 = pcl::console::find_switch (argc, argv, "-h");
  auto help_flag_2 = pcl::console::find_switch (argc, argv, "--help");

  if (help_flag_1 || help_flag_2) {
    printHelp (argc, argv);
    return -1;
  }

  if (argc > MAX_VALID_ARGS || argc < MIN_VALID_ARGS) {
    pcl::console::print_error ("Invalid number of arguments.\n");
    printHelp (argc, argv);
    return -1;
  }

  // Check if we are working with individual files
  auto const lmt_arg_indices = pcl::console::parse_file_extension_argument (argc, argv, ".lmt");
  auto const pcd_arg_indices = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");

  if (lmt_arg_indices.size () != 1 || pcd_arg_indices.size () != 1) {
    pcl::console::print_error ("Invalid number of arguments.\n");
    printHelp (argc, argv);
    return -1;
  }

  auto lmt_file = fs::path {argv[lmt_arg_indices.at (0)]};
  auto pcd_file = fs::path {argv[pcd_arg_indices.at (0)]};

  if (!checkValidFile (lmt_file)) {
    pcl::console::print_error ("A valid lmt file was not specified.\n");
    printHelp (argc, argv);
    return -1;
  }

  if (!checkValidFile (pcd_file)) {
    pcl::console::print_error ("A valid pcd file was not specified.\n");
    printHelp (argc, argv);
    return -1;
  }

  auto linemod = pcl::LINEMOD {};
  linemod.loadTemplates (lmt_file.c_str ());

  if (linemod.getNumOfTemplates () == 0) {
    pcl::console::print_error ("No valid templates found in lmt file.\n");
    printHelp (argc, argv);
    return -1;
  }

  // Load input pcd
  auto input_cloud = boost::make_shared <Cloud> ();
  if (pcl::io::loadPCDFile <PointType> (pcd_file.c_str (), *input_cloud) == -1) {
    pcl::console::print_error ("Failed to load: %s\n", pcd_file.c_str ());
    printHelp (argc, argv);
    return -1;
  }

  if (!input_cloud->isOrganized ()) {
    pcl::console::print_error ("Input cloud is not organised");
    printHelp (argc, argv);
    return -1;
  }

  auto matchedTemplates = detectTemplates (input_cloud, linemod);

  outputTemplateMatches (matchedTemplates);

  auto bestMatchTemplate = getClosestMatchedTemplate (matchedTemplates);

  outputTemplateMatch (bestMatchTemplate);

  auto const & multi_mod_template = linemod.getTemplate (bestMatchTemplate.template_id);

  auto x1 = bestMatchTemplate.x;
  auto y1 = bestMatchTemplate.y;
  auto x2 = x1 + multi_mod_template.region.width;
  auto y2 = y1 + multi_mod_template.region.height;

  auto image_extractor = pcl::io::PointCloudImageExtractorFromRGBField <pcl::PointXYZRGBA> {};
  auto image = boost::make_shared <pcl::PCLImage> ();
  auto extracted = image_extractor.extract (*input_cloud, *image);

  if (extracted) {
    for (int y = y1; y < y2; ++y)
      for (int x = x1; x < x2; ++x) {
        auto offset = 3 * (y*image->width + x);
        image->data.at (offset) = 255;
      }
  }

  pcl::visualization::ImageViewer image_viewer {"Image Viewer"};

  auto image_data = reinterpret_cast<unsigned char *>(image->data.data ());

  image_viewer.addRGBImage (image_data, image->width, image->height);
  image_viewer.spin ();
  return (0);
}