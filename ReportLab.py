from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image,  Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import io
from RobotLocalizationWithParticleFilters import *
import datetime
from PIL import ImageGrab

# Create a function to generate a plot as an image
def generate_plot(report):
    plt.plot(report.weights_mean_bn, '*r', ms=0.5, label="mean_weight_before_norm")
    plt.plot(report.best_weights, 'dg', ms=1, label="best_particle_weight")
    plt.plot(report.weights_mean_an, 'sb', ms=0.5, label="mean_weight_after_norm")
    plt.legend(loc="lower left")

    # Save the plot as an image in memory
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()

    return img_buffer

def generate_robot_position_plot(report):
    p1 = plt.scatter(report.robot_posxs, report.robot_posys, marker='+',
                     color='k', s=180, lw=3, label="Actual")
    p2 = plt.scatter(report.est_robot_posxs, report.est_robot_posys, marker='s', color='r', label="PF")
    plt.legend(loc="lower right")

    # Save the plot as an image in memory
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()

    return img_buffer


def create_report(report):
    # Create a PDF report using ReportLab
    cur_date = datetime.datetime.now()
    cur_date_str = str(cur_date.month) + str(cur_date.day) + str(cur_date.hour) + str(cur_date.minute) + str(cur_date.second)
    dir_name = os.path.split(report.path)[0]
    folder_name = os.path.basename(dir_name)
    pdf_filename = "Reports/reportlab_report"+cur_date_str+"_"+".pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=landscape(letter),
                            leftMargin=50,
                            rightMargin=50,
                            topMargin=50,
                            bottomMargin=50
                            )
    story = []

    # Add a title
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    story.append(Paragraph("PDF RobotLocalization Report", title_style))
    story.append(Spacer(1, 1))

    # Add text
    # text = "These are some information about the algorithm parameters:"
    # story.append(Paragraph(text, styles["Normal"]))
    # story.append(Spacer(1, 1))
    data = [
        ["Filename", report.path],
        ["Discarded", report.discarded],
        ["Samples num", report.sample_num],
        ["Number of particles", report.num_particles],
        ["Growth scale", report.growth_scale],
        ["Resample ratio(% num_of_particles)", report.resample_ratio],
        ["Number of resampling", report.resample_num],
        ["Velocity standard deviation", report.vel_std],
        ["Steer angle standard deviation", report.steer_std],
    ]

    # Create a Table object
    story.append(Table(data))
    story.append(Spacer(1, 1))

    # Add the generated plot as an image
    plot_image = Image(generate_plot(report), width=5*inch, height=3*inch, hAlign='LEFT')
    story.append(plot_image)
    story.append(Spacer(1, 1))

    # Add the generated plot as an image
    plot_image2 = Image(generate_robot_position_plot(report), width=3*inch, height=2*inch, hAlign='LEFT')
    story.append(plot_image2)
    story.append(Spacer(1, 1))

    # Capture a screenshot of the entire screen
    screenshot = ImageGrab.grab()
    screenshot.save("screenshot.png")
    img_width = 500
    img_height = 350
    screenshot_img = Image("screenshot.png", width=img_width, height=img_height, hAlign='CENTER')
    screenshot_img.vAligh='bottom'
    story.append(screenshot_img)

    # Build the PDF report
    doc.build(story)

    # Close the plot
    plt.close()

    print(f"PDF report saved as '{pdf_filename}'")

